// Function: sub_30750E0
// Address: 0x30750e0
//
__int64 __fastcall sub_30750E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // r12
  int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  bool v21; // si
  __int64 v22; // [rsp+10h] [rbp-120h]
  unsigned int v23; // [rsp+18h] [rbp-118h]
  _QWORD v24[2]; // [rsp+20h] [rbp-110h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-100h] BYREF
  _BYTE v26[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+60h] [rbp-D0h]
  unsigned __int64 v28[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v29[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+A0h] [rbp-90h]
  __int64 v31; // [rsp+A8h] [rbp-88h]
  __int16 v32; // [rsp+B0h] [rbp-80h]
  __int64 v33; // [rsp+B8h] [rbp-78h]
  void **v34; // [rsp+C0h] [rbp-70h]
  void **v35; // [rsp+C8h] [rbp-68h]
  __int64 v36; // [rsp+D0h] [rbp-60h]
  int v37; // [rsp+D8h] [rbp-58h]
  __int16 v38; // [rsp+DCh] [rbp-54h]
  char v39; // [rsp+DEh] [rbp-52h]
  __int64 v40; // [rsp+E0h] [rbp-50h]
  __int64 v41; // [rsp+E8h] [rbp-48h]
  void *v42; // [rsp+F0h] [rbp-40h] BYREF
  void *v43; // [rsp+F8h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v7 = *(_DWORD *)(v4 + 36);
  v22 = sub_3071AE0(v7);
  if ( !BYTE4(v22) )
    return 0;
  if ( v7 == 8170 )
  {
    v34 = &v42;
    v33 = sub_BD5C60(a2);
    v38 = 512;
    v28[1] = 0x200000000LL;
    v35 = &v43;
    v32 = 0;
    v43 = &unk_49DA0B0;
    v28[0] = (unsigned __int64)v29;
    v36 = 0;
    v37 = 0;
    v39 = 7;
    v40 = 0;
    v41 = 0;
    v30 = 0;
    v31 = 0;
    v42 = &unk_49DA100;
    sub_D5F1F0((__int64)v28, a2);
    v12 = *(_DWORD *)(a2 + 4);
    v27 = 257;
    v25[0] = a4;
    v25[1] = *(_QWORD *)(a2 + 32 * (1LL - (v12 & 0x7FFFFFF)));
    v24[0] = *(_QWORD *)(a4 + 8);
    v24[1] = v24[0];
    v10 = sub_B33D10((__int64)v28, 0x1FEAu, (__int64)v24, 2, (int)v25, 2, v23, (__int64)v26);
    nullsub_61();
    v42 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v28[0] != v29 )
      _libc_free(v28[0]);
    return v10;
  }
  if ( v7 - 8927 > 5 )
  {
    v28[0] = *(_QWORD *)(a4 + 8);
    v13 = (__int64 *)sub_B43CA0(a2);
    v14 = sub_B6E160(v13, v7, (__int64)v28, 1);
    v15 = *(_QWORD *)(a2 - 32) == 0;
    *(_QWORD *)(a2 + 80) = *(_QWORD *)(v14 + 24);
    if ( !v15 )
    {
      v16 = *(_QWORD *)(a2 - 24);
      **(_QWORD **)(a2 - 16) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(a2 - 16);
    }
    *(_QWORD *)(a2 - 32) = v14;
    v17 = *(_QWORD *)(v14 + 16);
    *(_QWORD *)(a2 - 24) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = a2 - 24;
    *(_QWORD *)(a2 - 16) = v14 + 16;
    *(_QWORD *)(v14 + 16) = a2 - 32;
    v18 = a2 + 32 * ((unsigned int)v22 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v18 )
    {
      v19 = *(_QWORD *)(v18 + 8);
      **(_QWORD **)(v18 + 16) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
    }
    *(_QWORD *)v18 = a4;
    v20 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v18 + 8) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v18 + 8;
    *(_QWORD *)(v18 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v18;
    return a2;
  }
  v8 = *(_QWORD *)(a4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = *(_DWORD *)(v8 + 8) >> 8;
  if ( !v9 || v9 == 101 )
    return 0;
  switch ( v7 )
  {
    case 0x22E0u:
      v21 = v9 == 1;
      break;
    case 0x22E1u:
      return 0;
    case 0x22E2u:
      v21 = v9 == 5;
      break;
    case 0x22E3u:
    case 0x22E4u:
      v21 = v9 == 3;
      break;
    default:
      v21 = v9 == 4;
      break;
  }
  return sub_AD64C0(*(_QWORD *)(a2 + 8), v21, 0);
}

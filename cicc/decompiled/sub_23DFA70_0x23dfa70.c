// Function: sub_23DFA70
// Address: 0x23dfa70
//
void __fastcall sub_23DFA70(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r9
  unsigned __int64 v5; // r10
  __int64 v6; // r11
  __int64 *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rax
  _QWORD *v10; // r8
  __int64 *v11; // rcx
  __int64 *v12; // rbx
  __int64 v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  _QWORD *v18; // [rsp+0h] [rbp-130h]
  unsigned __int64 v19; // [rsp+8h] [rbp-128h]
  __int64 v20; // [rsp+10h] [rbp-120h]
  _QWORD *v21; // [rsp+18h] [rbp-118h]
  __int64 *v22; // [rsp+20h] [rbp-110h]
  __int64 v23; // [rsp+28h] [rbp-108h]
  _QWORD v24[2]; // [rsp+30h] [rbp-100h] BYREF
  char v25; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+60h] [rbp-D0h]
  unsigned int *v27[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v28[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+A0h] [rbp-90h]
  __int64 v30; // [rsp+A8h] [rbp-88h]
  __int16 v31; // [rsp+B0h] [rbp-80h]
  __int64 v32; // [rsp+B8h] [rbp-78h]
  void **v33; // [rsp+C0h] [rbp-70h]
  void **v34; // [rsp+C8h] [rbp-68h]
  __int64 v35; // [rsp+D0h] [rbp-60h]
  int v36; // [rsp+D8h] [rbp-58h]
  __int16 v37; // [rsp+DCh] [rbp-54h]
  char v38; // [rsp+DEh] [rbp-52h]
  __int64 v39; // [rsp+E0h] [rbp-50h]
  __int64 v40; // [rsp+E8h] [rbp-48h]
  void *v41; // [rsp+F0h] [rbp-40h] BYREF
  void *v42; // [rsp+F8h] [rbp-38h] BYREF

  v33 = &v41;
  v32 = sub_BD5C60(a2);
  v37 = 512;
  v41 = &unk_49DA100;
  v27[0] = (unsigned int *)v28;
  v27[1] = (unsigned int *)0x200000000LL;
  v34 = &v42;
  v35 = 0;
  v36 = 0;
  v38 = 7;
  v39 = 0;
  v40 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v42 = &unk_49DA0B0;
  sub_D5F1F0((__int64)v27, a2);
  v4 = a1;
  if ( *(_BYTE *)a2 == 82 )
  {
    v5 = a1[20];
    v6 = a1[21];
  }
  else
  {
    v5 = a1[22];
    v6 = a1[23];
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a2 - 8);
  else
    v7 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *v7;
  v9 = v7[4];
  v10 = v24;
  v24[0] = v8;
  v11 = (__int64 *)&v25;
  v12 = v24;
  v24[1] = v9;
  if ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) == 14 )
    goto LABEL_8;
  while ( v11 != ++v12 )
  {
    while ( 1 )
    {
      v8 = *v12;
      if ( *(_BYTE *)(*(_QWORD *)(*v12 + 8) + 8LL) != 14 )
        break;
LABEL_8:
      v13 = v4[12];
      v18 = v10;
      ++v12;
      v19 = v5;
      v20 = v6;
      v26 = 257;
      v21 = v4;
      v22 = v11;
      v14 = sub_94BCF0(v27, v8, v13, (__int64)v11);
      v11 = v22;
      v10 = v18;
      *(v12 - 1) = (__int64)v14;
      v5 = v19;
      v6 = v20;
      v4 = v21;
      if ( v22 == v12 )
        goto LABEL_9;
    }
  }
LABEL_9:
  v26 = 257;
  v15 = sub_921880(v27, v5, v6, (int)v10, 2, (__int64)v11, 0);
  if ( *(_BYTE *)(a3 + 8) )
  {
    v17 = *(unsigned int *)(a3 + 24);
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 28) )
    {
      v23 = v15;
      sub_C8D5F0(a3 + 16, (const void *)(a3 + 32), v17 + 1, 8u, v17 + 1, v16);
      v17 = *(unsigned int *)(a3 + 24);
      v15 = v23;
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v17) = v15;
    ++*(_DWORD *)(a3 + 24);
  }
  nullsub_61();
  v41 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free((unsigned __int64)v27[0]);
}

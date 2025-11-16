// Function: sub_2713760
// Address: 0x2713760
//
__int64 __fastcall sub_2713760(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // [rsp+18h] [rbp-178h]
  _BYTE v10[16]; // [rsp+20h] [rbp-170h] BYREF
  __int64 v11; // [rsp+30h] [rbp-160h]
  char v12; // [rsp+40h] [rbp-150h]
  __int64 v13; // [rsp+50h] [rbp-140h]
  __int64 v14; // [rsp+58h] [rbp-138h]
  __int64 v15; // [rsp+60h] [rbp-130h]
  __int64 v16; // [rsp+68h] [rbp-128h] BYREF
  _BYTE *v17; // [rsp+70h] [rbp-120h]
  __int64 v18; // [rsp+78h] [rbp-118h]
  int v19; // [rsp+80h] [rbp-110h]
  char v20; // [rsp+84h] [rbp-10Ch]
  _BYTE v21[16]; // [rsp+88h] [rbp-108h] BYREF
  __int64 v22; // [rsp+98h] [rbp-F8h] BYREF
  _BYTE *v23; // [rsp+A0h] [rbp-F0h]
  __int64 v24; // [rsp+A8h] [rbp-E8h]
  int v25; // [rsp+B0h] [rbp-E0h]
  char v26; // [rsp+B4h] [rbp-DCh]
  _BYTE v27[16]; // [rsp+B8h] [rbp-D8h] BYREF
  char v28; // [rsp+C8h] [rbp-C8h]
  __int64 v29; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+D8h] [rbp-B8h] BYREF
  __int64 v31; // [rsp+E0h] [rbp-B0h]
  __int64 v32; // [rsp+E8h] [rbp-A8h]
  _BYTE v33[8]; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned __int64 v34; // [rsp+F8h] [rbp-98h]
  char v35; // [rsp+10Ch] [rbp-84h]
  _BYTE v36[16]; // [rsp+110h] [rbp-80h] BYREF
  _BYTE v37[8]; // [rsp+120h] [rbp-70h] BYREF
  unsigned __int64 v38; // [rsp+128h] [rbp-68h]
  char v39; // [rsp+13Ch] [rbp-54h]
  _BYTE v40[16]; // [rsp+140h] [rbp-50h] BYREF
  char v41; // [rsp+150h] [rbp-40h]

  v2 = *a2;
  v30 = 0;
  v29 = v2;
  sub_2712ED0((__int64)v10, (__int64)a1, &v29, &v30);
  if ( !v12 )
    return a1[4] + 136LL * *(_QWORD *)(v11 + 8) + 8;
  v9 = a1[5] - a1[4];
  v13 = 0;
  *(_QWORD *)(v11 + 8) = 0xF0F0F0F0F0F0F0F1LL * (v9 >> 3);
  v4 = *a2;
  v17 = v21;
  v29 = v4;
  LOWORD(v30) = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v18 = 2;
  v19 = 0;
  v20 = 1;
  v22 = 0;
  v23 = v27;
  v24 = 2;
  v25 = 0;
  v26 = 1;
  v28 = 0;
  BYTE2(v30) = 0;
  v31 = 0;
  v32 = 0;
  sub_C8CF70((__int64)v33, v36, 2, (__int64)v21, (__int64)&v16);
  sub_C8CF70((__int64)v37, v40, 2, (__int64)v27, (__int64)&v22);
  v8 = a1[5];
  v41 = v28;
  if ( v8 == a1[6] )
  {
    sub_2711E00(a1 + 4, v8, &v29, v5, v6, v7);
  }
  else
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = v29;
      *(_WORD *)(v8 + 8) = v30;
      *(_BYTE *)(v8 + 10) = BYTE2(v30);
      *(_WORD *)(v8 + 16) = v31;
      *(_QWORD *)(v8 + 24) = v32;
      sub_C8CF70(v8 + 32, (void *)(v8 + 64), 2, (__int64)v36, (__int64)v33);
      sub_C8CF70(v8 + 80, (void *)(v8 + 112), 2, (__int64)v40, (__int64)v37);
      *(_BYTE *)(v8 + 128) = v41;
      v8 = a1[5];
    }
    a1[5] = v8 + 136;
  }
  if ( !v39 )
    _libc_free(v38);
  if ( !v35 )
    _libc_free(v34);
  if ( !v26 )
    _libc_free((unsigned __int64)v23);
  if ( !v20 )
    _libc_free((unsigned __int64)v17);
  return a1[4] + v9 + 8;
}

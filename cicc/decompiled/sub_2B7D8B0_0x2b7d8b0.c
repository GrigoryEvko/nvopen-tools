// Function: sub_2B7D8B0
// Address: 0x2b7d8b0
//
__int64 ***__fastcall sub_2B7D8B0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rdx
  __int64 v7; // rcx
  _BYTE *v8; // rcx
  int v9; // edx
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned __int64 v12; // r8
  char *v13; // rcx
  int v14; // r10d
  char *v15; // rdx
  char *v16; // rsi
  unsigned int *v17; // rax
  __int64 *v18; // r9
  unsigned int *v19; // rdi
  __int64 v20; // rsi
  int v21; // edx
  __int64 ***v22; // r13
  __int64 v24; // [rsp+8h] [rbp-108h]
  int v25; // [rsp+8h] [rbp-108h]
  unsigned __int64 v26; // [rsp+10h] [rbp-100h]
  char *v27; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-E8h]
  _OWORD v29[3]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v30; // [rsp+60h] [rbp-B0h] BYREF
  char v31; // [rsp+68h] [rbp-A8h]
  _BYTE *v32; // [rsp+70h] [rbp-A0h]
  __int64 v33; // [rsp+78h] [rbp-98h]
  _BYTE v34[48]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v35; // [rsp+B0h] [rbp-60h]
  __int64 v36; // [rsp+B8h] [rbp-58h]
  _BYTE v37[16]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 *v38; // [rsp+D0h] [rbp-40h]
  __int64 *v39; // [rsp+D8h] [rbp-38h]

  v6 = a1[1];
  v7 = **a1;
  v32 = v34;
  v33 = 0xC00000000LL;
  v30 = v7;
  v38 = v6 + 421;
  v8 = *(_BYTE **)(a3 + 416);
  v39 = v6;
  v9 = *(_DWORD *)(a3 + 104);
  v31 = 0;
  v35 = v37;
  v36 = 0x200000000LL;
  if ( *v8 != 62 || v9 )
  {
    if ( v9 == 2 && *(_BYTE *)a1[2] )
    {
      v11 = 0;
      v10 = 0;
      v24 = a2;
      v27 = (char *)v29;
      v28 = 0xC00000000LL;
    }
    else
    {
      v24 = a2;
      v27 = (char *)v29;
      v10 = *(_DWORD *)(a3 + 152);
      v28 = 0xC00000000LL;
      v11 = *(_QWORD *)(a3 + 144);
    }
    sub_2B0FC00(v11, v10, (__int64)&v27, (__int64)v8, a2, (__int64)&v27);
    sub_2B7BF50((__int64)&v30, v24, v27, (unsigned int)v28);
    if ( v27 != (char *)v29 )
      _libc_free((unsigned __int64)v27);
  }
  else
  {
    sub_2B7BF50((__int64)&v30, a2, *(char **)(a3 + 144), *(unsigned int *)(a3 + 152));
  }
  v12 = *(unsigned int *)(a3 + 216);
  v27 = (char *)v29;
  v13 = (char *)v29;
  v28 = 0x300000000LL;
  v14 = v12;
  if ( v12 )
  {
    v15 = (char *)v29;
    if ( v12 > 3 )
    {
      v25 = v12;
      v26 = v12;
      sub_C8D5F0((__int64)&v27, v29, v12, 0x10u, v12, (__int64)&v27);
      v12 = v26;
      v13 = v27;
      v14 = v25;
      v15 = &v27[16 * (unsigned int)v28];
      v16 = &v27[16 * v26];
      if ( v16 != v15 )
        goto LABEL_10;
    }
    else
    {
      v16 = (char *)&v29[v12];
      if ( v16 != (char *)v29 )
      {
        do
        {
LABEL_10:
          if ( v15 )
          {
            *(_QWORD *)v15 = 0;
            *((_DWORD *)v15 + 2) = 0;
          }
          v15 += 16;
        }
        while ( v16 != v15 );
        v13 = v27;
      }
    }
    v17 = *(unsigned int **)(a3 + 208);
    v18 = a1[1];
    LODWORD(v28) = v14;
    v19 = &v17[2 * *(unsigned int *)(a3 + 216)];
    if ( v17 != v19 )
    {
      do
      {
        v20 = *v17;
        v17 += 2;
        v13 += 16;
        v21 = *(v17 - 1);
        *((_QWORD *)v13 - 2) = *(_QWORD *)(*v18 + 8 * v20);
        *((_DWORD *)v13 - 2) = v21;
      }
      while ( v19 != v17 );
      v13 = v27;
      v12 = (unsigned int)v28;
    }
  }
  v22 = sub_2B7B8F0((__int64)&v30, *(char **)(a3 + 112), *(unsigned int *)(a3 + 120), (__int64)v13, v12, 0, 0, 0, 0, 0);
  if ( v27 != (char *)v29 )
    _libc_free((unsigned __int64)v27);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  return v22;
}

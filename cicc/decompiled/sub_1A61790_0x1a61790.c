// Function: sub_1A61790
// Address: 0x1a61790
//
__int64 __fastcall sub_1A61790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v8; // r9
  __int64 *v9; // r10
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 *v13; // rdi
  __int64 *v14; // rax
  __int64 *v15; // rcx
  __int64 *v16; // r9
  __int64 v17; // rbx
  unsigned __int8 v18; // r12
  __int64 v19; // rdi
  unsigned __int8 v20; // al
  unsigned __int8 v22; // [rsp+Fh] [rbp-311h]
  __int64 v25; // [rsp+20h] [rbp-300h]
  __int64 v26; // [rsp+28h] [rbp-2F8h]
  __int64 *v27; // [rsp+28h] [rbp-2F8h]
  __int64 v28; // [rsp+30h] [rbp-2F0h] BYREF
  __int64 *v29; // [rsp+38h] [rbp-2E8h]
  __int64 *v30; // [rsp+40h] [rbp-2E0h]
  __int64 v31; // [rsp+48h] [rbp-2D8h]
  int v32; // [rsp+50h] [rbp-2D0h]
  _BYTE v33[136]; // [rsp+58h] [rbp-2C8h] BYREF
  _BYTE *v34; // [rsp+E0h] [rbp-240h] BYREF
  __int64 v35; // [rsp+E8h] [rbp-238h]
  _BYTE v36[560]; // [rsp+F0h] [rbp-230h] BYREF

  v34 = v36;
  v35 = 0x2000000000LL;
  sub_137D9B0(a1, (__int64)&v34);
  v8 = (__int64 *)v33;
  v28 = 0;
  v29 = (__int64 *)v33;
  v30 = (__int64 *)v33;
  v31 = 16;
  v32 = 0;
  if ( !(_DWORD)v35 )
    goto LABEL_15;
  v26 = a3;
  v9 = (__int64 *)v33;
  v10 = 0;
  v11 = 16LL * (unsigned int)v35;
  do
  {
LABEL_5:
    v12 = *(_QWORD *)&v34[v10 + 8];
    if ( v8 != v9 )
    {
LABEL_3:
      sub_16CCBA0((__int64)&v28, v12);
      v9 = v30;
      v8 = v29;
      goto LABEL_4;
    }
    v13 = &v8[HIDWORD(v31)];
    if ( v13 == v8 )
    {
LABEL_28:
      if ( HIDWORD(v31) >= (unsigned int)v31 )
        goto LABEL_3;
      ++HIDWORD(v31);
      *v13 = v12;
      v8 = v29;
      ++v28;
      v9 = v30;
    }
    else
    {
      v14 = v8;
      v15 = 0;
      while ( v12 != *v14 )
      {
        if ( *v14 == -2 )
          v15 = v14;
        if ( v13 == ++v14 )
        {
          if ( !v15 )
            goto LABEL_28;
          v10 += 16;
          *v15 = v12;
          v9 = v30;
          --v32;
          v8 = v29;
          ++v28;
          if ( v10 != v11 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
    }
LABEL_4:
    v10 += 16;
  }
  while ( v10 != v11 );
LABEL_14:
  a3 = v26;
LABEL_15:
  v22 = 0;
  v16 = &v28;
  v25 = a1 + 72;
  while ( 1 )
  {
    v17 = *(_QWORD *)(a1 + 80);
    if ( v17 == v25 )
      break;
    v18 = 0;
    do
    {
      v19 = v17 - 24;
      v27 = v16;
      v17 = *(_QWORD *)(v17 + 8);
      v20 = sub_1B5E140(v19, a2, a3, a4, a5);
      v16 = v27;
      if ( v20 )
        v18 = v20;
    }
    while ( v17 != v25 );
    if ( !v18 )
      break;
    v22 = v18;
  }
  if ( v30 != v29 )
    _libc_free((unsigned __int64)v30);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  return v22;
}

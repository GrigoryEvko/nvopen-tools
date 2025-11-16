// Function: sub_2582360
// Address: 0x2582360
//
__int64 __fastcall sub_2582360(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __m128i v4; // rax
  unsigned int v5; // r13d
  const void **v6; // r15
  unsigned __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // r12
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // r12
  const void **v17; // r15
  const void **v18; // rdx
  unsigned int v19; // ecx
  const void **v20; // [rsp+18h] [rbp-1E8h]
  char v21; // [rsp+2Fh] [rbp-1D1h] BYREF
  __int64 v22; // [rsp+30h] [rbp-1D0h] BYREF
  int v23; // [rsp+38h] [rbp-1C8h]
  __m128i v24; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v25; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v26; // [rsp+58h] [rbp-1A8h]
  __int64 v27; // [rsp+60h] [rbp-1A0h]
  __int64 v28; // [rsp+68h] [rbp-198h]
  const void **v29; // [rsp+70h] [rbp-190h]
  __int64 v30; // [rsp+78h] [rbp-188h]
  _BYTE v31[128]; // [rsp+80h] [rbp-180h] BYREF
  _QWORD v32[4]; // [rsp+100h] [rbp-100h] BYREF
  __int64 v33; // [rsp+120h] [rbp-E0h]
  unsigned int v34; // [rsp+130h] [rbp-D0h]
  unsigned __int64 v35; // [rsp+138h] [rbp-C8h]
  unsigned int v36; // [rsp+140h] [rbp-C0h]
  char v37; // [rsp+148h] [rbp-B8h] BYREF

  sub_2560F70((__int64)v32, a1 + 88);
  v29 = (const void **)v31;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v30 = 0x800000000LL;
  v4.m128i_i64[0] = sub_250D2C0(a3, 0);
  v24 = v4;
  if ( (unsigned __int8)sub_2580850(a1, a2, &v24, &v25, &v21, 1) )
  {
    if ( v21 )
    {
      *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
    }
    else
    {
      v17 = v29;
      v18 = &v29[2 * (unsigned int)v30];
      if ( v18 != v29 )
      {
        do
        {
          if ( *(_BYTE *)(a1 + 105) )
          {
            v20 = v18;
            sub_2575FB0((_DWORD *)(a1 + 112), v17);
            v19 = *(_DWORD *)(a1 + 152);
            v18 = v20;
            if ( v19 >= unk_4FEF868 )
              *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
            else
              *(_BYTE *)(a1 + 288) &= v19 == 0;
          }
          v17 += 2;
        }
        while ( v18 != v17 );
      }
    }
    v5 = (unsigned __int8)sub_255BE50((__int64)v32, (const void ***)(a1 + 88));
  }
  else
  {
    v5 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
  v6 = v29;
  v7 = (unsigned __int64)&v29[2 * (unsigned int)v30];
  if ( v29 != (const void **)v7 )
  {
    do
    {
      v7 -= 16LL;
      if ( *(_DWORD *)(v7 + 8) > 0x40u && *(_QWORD *)v7 )
        j_j___libc_free_0_0(*(_QWORD *)v7);
    }
    while ( v6 != (const void **)v7 );
    v7 = (unsigned __int64)v29;
  }
  if ( (_BYTE *)v7 != v31 )
    _libc_free(v7);
  v8 = (unsigned int)v28;
  if ( (_DWORD)v28 )
  {
    v9 = v26;
    v23 = 0;
    v22 = -1;
    v24.m128i_i32[2] = 0;
    v10 = v26 + 16LL * (unsigned int)v28;
    v24.m128i_i64[0] = -2;
    do
    {
      if ( *(_DWORD *)(v9 + 8) > 0x40u && *(_QWORD *)v9 )
        j_j___libc_free_0_0(*(_QWORD *)v9);
      v9 += 16;
    }
    while ( v10 != v9 );
    sub_969240(v24.m128i_i64);
    sub_969240(&v22);
    v8 = (unsigned int)v28;
  }
  sub_C7D6A0(v26, 16 * v8, 8);
  v11 = v35;
  v12 = v35 + 16LL * v36;
  v32[0] = &unk_4A170B8;
  if ( v35 != v12 )
  {
    do
    {
      v12 -= 16LL;
      if ( *(_DWORD *)(v12 + 8) > 0x40u && *(_QWORD *)v12 )
        j_j___libc_free_0_0(*(_QWORD *)v12);
    }
    while ( v11 != v12 );
    v12 = v35;
  }
  if ( (char *)v12 != &v37 )
    _libc_free(v12);
  v13 = v34;
  if ( v34 )
  {
    v14 = v33;
    v15 = v33 + 16LL * v34;
    do
    {
      if ( *(_DWORD *)(v14 + 8) > 0x40u && *(_QWORD *)v14 )
        j_j___libc_free_0_0(*(_QWORD *)v14);
      v14 += 16;
    }
    while ( v15 != v14 );
    v13 = v34;
  }
  sub_C7D6A0(v33, 16 * v13, 8);
  return v5;
}

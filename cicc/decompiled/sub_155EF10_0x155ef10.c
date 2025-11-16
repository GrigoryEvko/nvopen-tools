// Function: sub_155EF10
// Address: 0x155ef10
//
__int64 __fastcall sub_155EF10(__int64 *a1, const void *a2, __int64 a3)
{
  __int64 v3; // r13
  size_t v4; // r12
  __int64 v5; // rbx
  __int64 *v6; // rdi
  __int64 v7; // r12
  __int64 *v8; // r14
  unsigned __int64 v9; // rax
  __int64 *v10; // r13
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 *v14; // r13
  char *v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // [rsp+10h] [rbp-130h]
  __int64 v19; // [rsp+28h] [rbp-118h] BYREF
  __int64 *v20; // [rsp+30h] [rbp-110h] BYREF
  __int64 v21; // [rsp+38h] [rbp-108h]
  _BYTE v22[64]; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v23[2]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE v24[176]; // [rsp+90h] [rbp-B0h] BYREF

  v3 = 0;
  if ( !a3 )
    return v3;
  v4 = 8 * a3;
  v18 = *a1;
  v5 = (8 * a3) >> 3;
  v23[0] = (unsigned __int64)v24;
  v23[1] = 0x2000000000LL;
  v20 = (__int64 *)v22;
  v21 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_16CD150(&v20, v22, (8 * a3) >> 3, 8);
    v6 = &v20[(unsigned int)v21];
    goto LABEL_19;
  }
  v6 = (__int64 *)v22;
  if ( v4 )
  {
LABEL_19:
    memcpy(v6, a2, v4);
    v6 = v20;
    LODWORD(v4) = v21;
  }
  LODWORD(v21) = v5 + v4;
  v7 = (unsigned int)(v5 + v4);
  v8 = &v6[v7];
  if ( &v6[v7] != v6 )
  {
    _BitScanReverse64(&v9, (v7 * 8) >> 3);
    sub_155EB40(v6, &v6[v7], 2LL * (int)(63 - (v9 ^ 0x3F)));
    if ( (unsigned __int64)v7 <= 16 )
    {
      sub_155ED50(v6, v8);
    }
    else
    {
      v10 = v6 + 16;
      sub_155ED50(v6, v6 + 16);
      if ( v8 != v6 + 16 )
      {
        do
        {
          v11 = v10 - 1;
          v19 = *v10;
          while ( sub_155E9A0(&v19, *v11) )
          {
            v12 = *v11--;
            v11[2] = v12;
          }
          ++v10;
          v11[1] = v19;
        }
        while ( v8 != v10 );
      }
    }
    v14 = v20;
    v15 = (char *)&v20[(unsigned int)v21];
    if ( v20 != (__int64 *)v15 )
    {
      do
      {
        v16 = *v14++;
        sub_16BD4C0(v23, v16);
      }
      while ( v15 != (char *)v14 );
    }
  }
  v3 = sub_16BDDE0(v18 + 248, v23, &v19);
  if ( !v3 )
  {
    v17 = sub_22077B0(8LL * (unsigned int)v21 + 24);
    v3 = v17;
    if ( v17 )
      sub_155EE60(v17, v20, (unsigned int)v21);
    sub_16BDA20(v18 + 248, v3, v19);
  }
  if ( v20 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v20);
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
  return v3;
}

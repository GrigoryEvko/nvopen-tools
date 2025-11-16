// Function: sub_16BA290
// Address: 0x16ba290
//
__int64 __fastcall sub_16BA290(char *src, signed __int64 n, _QWORD *a3)
{
  __int16 v3; // ax
  char *v5; // rdx
  char *v6; // r15
  __int16 *v7; // r14
  __int64 result; // rax
  __int64 v9; // rbx
  _BYTE *v10; // rax
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rax
  __int16 *v14; // rax
  unsigned __int8 v15; // [rsp-58h] [rbp-58h]
  int v16; // [rsp-58h] [rbp-58h]
  signed __int64 v17; // [rsp-50h] [rbp-50h]
  char *v18; // [rsp-48h] [rbp-48h] BYREF
  _BYTE *v19; // [rsp-40h] [rbp-40h] BYREF

  if ( (n & 1) != 0 )
    return 0;
  if ( !n )
    return 1;
  v3 = *(_WORD *)src;
  v18 = src;
  v17 = 0;
  v5 = src;
  v6 = &src[n];
  v7 = 0;
  if ( v3 == -2 )
  {
    if ( v6 == src )
    {
      v6 = 0;
    }
    else
    {
      if ( (unsigned __int64)n > 0x7FFFFFFFFFFFFFFELL )
        sub_4262D8((__int64)"vector::_M_range_insert");
      v17 = n;
      v16 = n >> 1;
      v7 = (__int16 *)sub_22077B0(n);
      v14 = (__int16 *)memcpy(v7, src, n);
      v6 = (char *)v7 + n;
      if ( v16 )
      {
        do
        {
          *v14 = __ROL2__(*v14, 8);
          ++v14;
        }
        while ( &v7[v16] != v14 );
      }
    }
    v18 = (char *)v7;
    v3 = *v7;
    v5 = (char *)v7;
  }
  if ( v3 == -257 )
    v18 = v5 + 2;
  sub_22410F0(a3, 4 * n + 1, 0);
  v19 = (_BYTE *)*a3;
  if ( (unsigned int)sub_16F0B80(&v18, v6, &v19, &v19[a3[1]], 0) )
  {
    a3[1] = 0;
    *(_BYTE *)*a3 = 0;
    result = 0;
  }
  else
  {
    sub_22410F0(a3, &v19[-*a3], 0);
    v9 = a3[1];
    v10 = (_BYTE *)*a3;
    v11 = v9 + 1;
    if ( (_QWORD *)*a3 == a3 + 2 )
      v12 = 15;
    else
      v12 = a3[2];
    if ( v11 > v12 )
    {
      sub_2240BB0(a3, a3[1], 0, 0, 1);
      v10 = (_BYTE *)*a3;
    }
    v10[v9] = 0;
    v13 = (_BYTE *)*a3;
    a3[1] = v11;
    v13[v9 + 1] = 0;
    sub_2240CE0(a3, a3[1] - 1LL, 1);
    result = 1;
  }
  if ( v7 )
  {
    v15 = result;
    j_j___libc_free_0(v7, v17);
    return v15;
  }
  return result;
}

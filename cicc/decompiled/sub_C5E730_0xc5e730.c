// Function: sub_C5E730
// Address: 0xc5e730
//
__int64 __fastcall sub_C5E730(char *src, size_t n, _QWORD *a3)
{
  __int16 v3; // ax
  char *v5; // rdx
  char *v6; // r9
  size_t v7; // r15
  __int16 *v8; // r14
  __int64 result; // rax
  __int64 v10; // rbx
  _BYTE *v11; // rax
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdx
  _BYTE *v14; // rax
  char *v15; // rax
  char *v16; // [rsp-50h] [rbp-50h]
  unsigned __int8 v17; // [rsp-50h] [rbp-50h]
  char *v18; // [rsp-48h] [rbp-48h] BYREF
  _BYTE *v19; // [rsp-40h] [rbp-40h] BYREF

  if ( (n & 1) != 0 )
    return 0;
  if ( !n )
    return 1;
  v3 = *(_WORD *)src;
  v18 = src;
  v5 = src;
  v6 = &src[n];
  v7 = 0;
  v8 = 0;
  if ( v3 == -2 )
  {
    if ( src == v6 )
    {
      v6 = 0;
    }
    else
    {
      if ( n > 0x7FFFFFFFFFFFFFFELL )
        sub_4262D8((__int64)"vector::_M_range_insert");
      v7 = n;
      v8 = (__int16 *)sub_22077B0(n);
      v15 = (char *)memcpy(v8, src, n);
      v6 = (char *)v8 + n;
      if ( (__int16 *)((char *)v8 + n) != v8 )
      {
        do
        {
          *(_WORD *)v15 = __ROL2__(*(_WORD *)v15, 8);
          v15 += 2;
        }
        while ( v6 != v15 );
      }
    }
    v18 = (char *)v8;
    v3 = *v8;
    v5 = (char *)v8;
  }
  if ( v3 == -257 )
    v18 = v5 + 2;
  v16 = v6;
  sub_22410F0(a3, 4 * n + 1, 0);
  v19 = (_BYTE *)*a3;
  if ( (unsigned int)sub_F03400(&v18, v16, &v19, &v19[a3[1]], 0) )
  {
    a3[1] = 0;
    *(_BYTE *)*a3 = 0;
    result = 0;
  }
  else
  {
    sub_22410F0(a3, &v19[-*a3], 0);
    v10 = a3[1];
    v11 = (_BYTE *)*a3;
    v12 = v10 + 1;
    if ( (_QWORD *)*a3 == a3 + 2 )
      v13 = 15;
    else
      v13 = a3[2];
    if ( v12 > v13 )
    {
      sub_2240BB0(a3, a3[1], 0, 0, 1);
      v11 = (_BYTE *)*a3;
    }
    v11[v10] = 0;
    v14 = (_BYTE *)*a3;
    a3[1] = v12;
    v14[v10 + 1] = 0;
    sub_2240CE0(a3, a3[1] - 1LL, 1);
    result = 1;
  }
  if ( v8 )
  {
    v17 = result;
    j_j___libc_free_0(v8, v7);
    return v17;
  }
  return result;
}

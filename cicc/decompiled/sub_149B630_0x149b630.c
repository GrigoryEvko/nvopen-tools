// Function: sub_149B630
// Address: 0x149b630
//
__int64 __fastcall sub_149B630(__int64 a1, _BYTE *a2, __int64 a3, _DWORD *a4)
{
  unsigned int v4; // r12d
  _BYTE *v5; // rax
  size_t v6; // rdx
  __int64 *v8; // r15
  const void *v9; // r14
  size_t v10; // r8
  __int64 i; // r12
  int v12; // eax
  __int64 v13; // rcx
  __int64 *v14; // rbx
  size_t v15; // r13
  size_t n; // [rsp+10h] [rbp-40h]
  size_t v18; // [rsp+18h] [rbp-38h]

  v4 = 0;
  v5 = sub_149A7F0(a2, a3);
  if ( !v6 )
    return v4;
  v8 = qword_4F9B700;
  v9 = v5;
  v10 = v6;
  for ( i = 422; i > 0; i = i - v13 - 1 )
  {
    while ( 1 )
    {
      v13 = i >> 1;
      v14 = &v8[2 * (i >> 1)];
      v15 = v14[1];
      if ( v15 <= v10 )
        break;
      v18 = v10;
      v12 = memcmp((const void *)*v14, v9, v10);
      v10 = v18;
      v13 = i >> 1;
      if ( v12 )
        goto LABEL_16;
LABEL_10:
      if ( v15 < v10 )
        goto LABEL_11;
LABEL_7:
      i = v13;
      if ( v13 <= 0 )
        goto LABEL_12;
    }
    if ( !v15 )
      goto LABEL_11;
    n = v10;
    v12 = memcmp((const void *)*v14, v9, v14[1]);
    v13 = i >> 1;
    v10 = n;
    if ( !v12 )
    {
      if ( v15 == n )
        goto LABEL_7;
      goto LABEL_10;
    }
LABEL_16:
    if ( v12 >= 0 )
      goto LABEL_7;
LABEL_11:
    v8 = v14 + 2;
  }
LABEL_12:
  v4 = 0;
  if ( v8 != &qword_4F9D160 && v8[1] == v10 && !memcmp((const void *)*v8, v9, v10) )
  {
    v4 = 1;
    *a4 = ((char *)v8 - (char *)qword_4F9B700) >> 4;
  }
  return v4;
}

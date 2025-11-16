// Function: sub_155FA70
// Address: 0x155fa70
//
__int64 __fastcall sub_155FA70(__int64 *a1, int *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  int v6; // eax
  unsigned int v7; // eax
  __int64 *v8; // rsi
  __int64 v9; // r12
  __int64 *v10; // rax
  int *v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // [rsp-70h] [rbp-70h]
  unsigned int v15; // [rsp-70h] [rbp-70h]
  __int64 *v16; // [rsp-68h] [rbp-68h] BYREF
  __int64 v17; // [rsp-60h] [rbp-60h]
  __int64 v18; // [rsp-58h] [rbp-58h] BYREF
  __int64 v19; // [rsp-50h] [rbp-50h] BYREF

  if ( !a3 )
    return 0;
  v4 = 4 * a3;
  v6 = a2[4 * a3 - 4];
  if ( v6 == -1 && (a3 == 1 || (v6 = a2[v4 - 8], v6 == -1)) )
  {
    v17 = 0x400000001LL;
    v8 = &v18;
    v10 = &v19;
    v16 = &v18;
  }
  else
  {
    v16 = &v18;
    v7 = v6 + 2;
    v8 = &v18;
    v17 = 0x400000000LL;
    v9 = v7;
    if ( v7 > 4uLL )
    {
      v15 = v7;
      sub_16CD150(&v16, &v18, v7, 8);
      v8 = v16;
      v7 = v15;
    }
    LODWORD(v17) = v7;
    v10 = &v8[v9];
    if ( &v8[v9] == v8 )
      goto LABEL_11;
  }
  do
  {
    if ( v8 )
      *v8 = 0;
    ++v8;
  }
  while ( v8 != v10 );
  v8 = v16;
LABEL_11:
  if ( &a2[v4] != a2 )
  {
    v11 = a2;
    do
    {
      v12 = *v11;
      v13 = *((_QWORD *)v11 + 1);
      v11 += 4;
      v8[v12 + 1] = v13;
      v8 = v16;
    }
    while ( v11 != &a2[v4] );
  }
  result = sub_155F990(a1, v8, (unsigned int)v17);
  if ( v16 != &v18 )
  {
    v14 = result;
    _libc_free((unsigned __int64)v16);
    return v14;
  }
  return result;
}

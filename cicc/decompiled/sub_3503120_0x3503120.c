// Function: sub_3503120
// Address: 0x3503120
//
__int64 __fastcall sub_3503120(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 (*v6)(void); // rax
  __int64 *v7; // r13
  __int64 *v8; // r15
  __int64 *v9; // rbx
  __int64 result; // rax

  v6 = *(unsigned __int8 (**)(void))(*a1 + 64LL);
  if ( (char *)v6 != (char *)sub_3502E20 )
  {
    if ( !v6() )
    {
      v8 = (__int64 *)a1[6];
      v7 = (__int64 *)a1[7];
      goto LABEL_3;
    }
    return 0;
  }
  v7 = (__int64 *)a1[7];
  v8 = (__int64 *)a1[6];
  if ( v7 == v8 )
    return 0;
LABEL_3:
  v9 = v8 + 1;
  if ( v7 != v8 + 1 )
  {
    do
    {
      if ( sub_3502FE0((__int64)(a1 + 9), *v8, *v9, a4, a5, a6) )
        v8 = v9;
      ++v9;
    }
    while ( v7 != v9 );
    v9 = (__int64 *)a1[7];
  }
  result = *v8;
  if ( v8 != v9 - 1 )
  {
    *v8 = *(v9 - 1);
    *(v9 - 1) = result;
    v8 = (__int64 *)(a1[7] - 8LL);
  }
  a1[7] = v8;
  return result;
}

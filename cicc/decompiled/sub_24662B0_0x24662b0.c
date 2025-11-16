// Function: sub_24662B0
// Address: 0x24662b0
//
unsigned __int64 __fastcall sub_24662B0(
        __int64 a1,
        __int64 *a2,
        int *a3,
        __int64 a4,
        char a5,
        char a6,
        unsigned __int64 a7)
{
  int *v7; // rbx
  int v9; // r14d
  int *v10; // r12
  int v11; // eax
  int v12; // ecx

  v7 = a3;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 168LL) )
  {
    v9 = a5 == 0 ? 79 : 54;
  }
  else
  {
    if ( !*(_BYTE *)(*(_QWORD *)a1 + 170LL) )
      goto LABEL_3;
    v9 = 54;
  }
  v10 = &a3[a4];
  if ( v10 != a3 )
  {
    do
    {
      v11 = *v7++;
      a7 = sub_A7A090((__int64 *)&a7, a2, v11 + 1, v9);
    }
    while ( v10 != v7 );
  }
LABEL_3:
  if ( a6 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 169LL) )
    {
      v12 = a5 == 0 ? 79 : 54;
    }
    else
    {
      if ( !*(_BYTE *)(*(_QWORD *)a1 + 171LL) )
        return a7;
      v12 = 54;
    }
    return sub_A7A090((__int64 *)&a7, a2, 0, v12);
  }
  return a7;
}

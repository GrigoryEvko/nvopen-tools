// Function: sub_29136F0
// Address: 0x29136f0
//
__int64 __fastcall sub_29136F0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8,
        char a9,
        __int64 a10,
        unsigned __int64 a11,
        char a12)
{
  __int64 v13; // rax
  unsigned __int64 v14; // r9
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  unsigned int v17; // r8d
  char v19; // dl

  v13 = a10;
  v14 = a11;
  if ( a9 )
  {
    if ( a7 <= a3 )
      a3 = a7;
    *a4 = a3;
    a4[1] = a8 + a2;
    if ( a12 )
    {
LABEL_5:
      v15 = *a4;
      v16 = a4[1];
      if ( *a4 == a10 && v16 == a11 )
        return 0;
      v17 = 2;
      if ( v16 < a11 )
        return v17;
      goto LABEL_7;
    }
  }
  else
  {
    *a4 = a3;
    a4[1] = a2;
    if ( a12 )
      goto LABEL_5;
  }
  v13 = sub_AF3FE0(a1);
  if ( !v19 )
    return 0;
  if ( v13 == *a4 )
  {
    v17 = 1;
    if ( !a4[1] )
      return v17;
    v16 = a4[1];
    v15 = v13;
  }
  else
  {
    v15 = *a4;
    v16 = a4[1];
  }
  v14 = 0;
LABEL_7:
  v17 = 2;
  if ( v14 + v13 >= v15 + v16 )
    return 0;
  return v17;
}

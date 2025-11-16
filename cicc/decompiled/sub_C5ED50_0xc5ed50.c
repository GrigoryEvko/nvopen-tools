// Function: sub_C5ED50
// Address: 0xc5ed50
//
__int64 __fastcall sub_C5ED50(__int64 *a1, unsigned __int64 *a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  __int64 v10; // r8

  if ( a4 )
  {
    v7 = *a4 & 0xFFFFFFFFFFFFFFFELL;
    if ( v7 )
    {
      v10 = 0;
LABEL_11:
      *a4 = v7 | 1;
      return v10;
    }
    *a4 = 0;
  }
  if ( (unsigned __int8)sub_C5EA20((__int64)a1, *a2, a3, a4, a5) )
  {
    v8 = *a1;
    v9 = a1[1];
    if ( *a2 <= v9 )
      v9 = *a2;
    *a2 += a3;
    v10 = v9 + v8;
  }
  else
  {
    v10 = 0;
  }
  if ( !a4 )
    return v10;
  v7 = *a4 & 0xFFFFFFFFFFFFFFFELL;
  if ( v7 )
    goto LABEL_11;
  *a4 = 1;
  return v10;
}

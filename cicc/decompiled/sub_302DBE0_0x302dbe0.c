// Function: sub_302DBE0
// Address: 0x302dbe0
//
__int64 __fastcall sub_302DBE0(_BYTE **a1, __int64 a2, unsigned int a3, _BYTE *a4, __int64 a5)
{
  if ( !a4 || !*a4 )
    goto LABEL_5;
  if ( a4[1] )
    return 1;
  if ( *a4 == 114 )
  {
LABEL_5:
    sub_3027FE0(a1, a2, a3, a5, a5);
    return 0;
  }
  return sub_31F1C50();
}

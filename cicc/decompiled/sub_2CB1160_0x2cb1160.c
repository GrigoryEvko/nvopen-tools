// Function: sub_2CB1160
// Address: 0x2cb1160
//
char __fastcall sub_2CB1160(unsigned __int64 a1, __int64 a2, __int64 a3, int *a4, unsigned int a5, __int64 a6, int *a7)
{
  unsigned __int8 *v9; // r13
  unsigned __int8 *v10; // r14
  char result; // al

  v9 = *(unsigned __int8 **)(a2 - 64);
  v10 = *(unsigned __int8 **)(a2 - 32);
  if ( !sub_2CAFF80(a1, (__int64)v9, a5, *a4) || (result = sub_2CB0FF0(a1, v9, a4, (__int64)v10, a7, a3, a6)) == 0 )
  {
    result = sub_2CAFF80(a1, (__int64)v10, a5, *a4);
    if ( result )
      return sub_2CB0FF0(a1, v10, a4, (__int64)v9, a7, a3, a6);
  }
  return result;
}

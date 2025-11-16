// Function: sub_16F4F50
// Address: 0x16f4f50
//
__int64 __fastcall sub_16F4F50(__int64 a1, signed __int64 a2, size_t a3, int a4)
{
  if ( a2 >= 0 )
    return sub_16F4D80(a1, a2, a3, a4, 0);
  else
    return sub_16F4D80(a1, -a2, a3, a4, 1);
}

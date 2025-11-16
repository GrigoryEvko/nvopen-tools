// Function: sub_9B6140
// Address: 0x9b6140
//
__int64 __fastcall sub_9B6140(__int64 a1, unsigned int a2, __m128i *a3, unsigned __int8 *a4, unsigned __int8 *a5)
{
  if ( (unsigned __int8)sub_985700(a4, a5)
    || *a4 <= 0x15u && (unsigned __int8)sub_AC30F0(a4) && (unsigned __int8)sub_9A6530((__int64)a5, a1, a3, a2) )
  {
    return 1;
  }
  else
  {
    return sub_9B5220(a4, a5, a1, a2, a3);
  }
}

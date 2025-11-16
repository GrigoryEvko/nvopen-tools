// Function: sub_2E4F140
// Address: 0x2e4f140
//
char __fastcall sub_2E4F140(__int64 a1, __int64 a2)
{
  if ( (unsigned __int64)(a2 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(a1 - 1) > 0xFFFFFFFFFFFFFFFDLL )
    return a2 == a1;
  else
    return sub_2E88AF0(a1, a2, 3);
}

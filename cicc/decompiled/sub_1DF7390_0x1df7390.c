// Function: sub_1DF7390
// Address: 0x1df7390
//
char __fastcall sub_1DF7390(__int64 a1, __int64 a2)
{
  if ( (unsigned __int64)(a2 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(a1 - 1) > 0xFFFFFFFFFFFFFFFDLL )
    return a2 == a1;
  else
    return sub_1E15D60(a1, a2, 3);
}

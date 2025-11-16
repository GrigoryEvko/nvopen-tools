// Function: sub_134AC50
// Address: 0x134ac50
//
int __fastcall sub_134AC50(__int64 a1, __int64 a2)
{
  int result; // eax

  sub_133DF70(a1, a2 + 80);
  sub_133DF70(a1, a2 + 19520);
  result = sub_133DF70(a1, a2 + 38960);
  if ( *(_BYTE *)(a2 + 17) )
    return sub_1348950(a1, a2 + 62384);
  return result;
}

// Function: sub_2BF0A20
// Address: 0x2bf0a20
//
bool __fastcall sub_2BF0A20(__int64 a1)
{
  bool result; // al
  __int64 v3; // rdi

  result = 0;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 )
    return a1 == sub_2BF0500(v3);
  return result;
}

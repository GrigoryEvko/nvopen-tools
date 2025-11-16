// Function: sub_1642E20
// Address: 0x1642e20
//
bool __fastcall sub_1642E20(__int64 a1)
{
  bool result; // al
  __int64 v2[2]; // [rsp+8h] [rbp-18h] BYREF

  v2[0] = a1;
  result = sub_155D850(v2, "statepoint-id", 0xDu);
  if ( !result )
    return sub_155D850(v2, "statepoint-num-patch-bytes", 0x1Au);
  return result;
}

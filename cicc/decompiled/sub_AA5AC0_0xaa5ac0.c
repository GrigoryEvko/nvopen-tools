// Function: sub_AA5AC0
// Address: 0xaa5ac0
//
bool __fastcall sub_AA5AC0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // ecx
  bool result; // al
  unsigned int v4; // ecx

  v1 = sub_AA4FF0(a1);
  if ( !v1 )
    BUG();
  v2 = *(unsigned __int8 *)(v1 - 24);
  result = 1;
  if ( (_BYTE)v2 != 95 )
  {
    v4 = v2 - 39;
    if ( v4 <= 0x38 )
      return ((1LL << v4) & 0x100060000000001LL) == 0;
  }
  return result;
}

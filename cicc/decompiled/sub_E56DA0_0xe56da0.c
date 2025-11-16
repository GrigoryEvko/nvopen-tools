// Function: sub_E56DA0
// Address: 0xe56da0
//
__int64 __fastcall sub_E56DA0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 312);
  if ( !*(_BYTE *)(result + 21) )
    return sub_E9A100();
  return result;
}

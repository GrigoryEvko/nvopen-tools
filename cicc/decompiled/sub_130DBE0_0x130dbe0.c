// Function: sub_130DBE0
// Address: 0x130dbe0
//
__int64 __fastcall sub_130DBE0(__int64 a1)
{
  __int64 result; // rax

  result = -1;
  if ( a1 != -1 )
    result = (1LL << a1) - 1;
  unk_4C6F120 = result;
  return result;
}

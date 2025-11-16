// Function: sub_91B6C0
// Address: 0x91b6c0
//
__int64 __fastcall sub_91B6C0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 136);
  if ( !result )
    return *(_QWORD *)(a1 + 8);
  return result;
}

// Function: sub_9227D0
// Address: 0x9227d0
//
__int64 __fastcall sub_9227D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_9439D0();
  if ( !result )
    sub_91B8A0("could not lookup variable in map!", (_DWORD *)(a2 + 64), 1);
  return result;
}

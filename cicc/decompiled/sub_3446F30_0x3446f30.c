// Function: sub_3446F30
// Address: 0x3446f30
//
__int64 __fastcall sub_3446F30(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // eax

  result = sub_23CF1F0(*(_DWORD **)(a1 + 8), *(_BYTE **)(a2 + 96));
  if ( (_BYTE)result )
  {
    LOBYTE(v3) = sub_3446F00(a1);
    return v3 ^ 1u;
  }
  return result;
}

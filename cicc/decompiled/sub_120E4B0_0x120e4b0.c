// Function: sub_120E4B0
// Address: 0x120e4b0
//
__int64 __fastcall sub_120E4B0(__int64 a1, unsigned int a2, _BYTE *a3, _DWORD *a4)
{
  __int64 result; // rax

  if ( !(_BYTE)a2 )
    return a2;
  result = sub_120E240(a1, a3);
  if ( !(_BYTE)result )
    return sub_120E3E0(a1, a4);
  return result;
}

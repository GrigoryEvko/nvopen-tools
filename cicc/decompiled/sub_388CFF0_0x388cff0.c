// Function: sub_388CFF0
// Address: 0x388cff0
//
__int64 __fastcall sub_388CFF0(__int64 a1, unsigned int a2, _BYTE *a3, _DWORD *a4)
{
  __int64 result; // rax

  if ( !(_BYTE)a2 )
    return a2;
  result = sub_388CDD0(a1, a3);
  if ( !(_BYTE)result )
    return sub_388CF30(a1, a4);
  return result;
}

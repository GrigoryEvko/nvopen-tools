// Function: sub_1C98880
// Address: 0x1c98880
//
__int64 __fastcall sub_1C98880(__int64 a1, unsigned int a2, _DWORD *a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r13d
  __int64 result; // rax

  v4 = sub_1C30260(a2);
  if ( !(_BYTE)v4 )
  {
    v5 = v4;
    if ( (!byte_4FBE460 || !sub_1C98810(a1, a2)) && (a2 - 133 > 4 || ((1LL << ((unsigned __int8)a2 + 123)) & 0x15) == 0) )
      return v5;
    goto LABEL_5;
  }
  LOBYTE(result) = sub_1C302A0(a2);
  if ( !(_BYTE)result )
  {
LABEL_5:
    *a3 = 0;
    return 1;
  }
  *a3 = 1;
  return (unsigned int)result;
}

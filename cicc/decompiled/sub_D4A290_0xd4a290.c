// Function: sub_D4A290
// Address: 0xd4a290
//
__int64 __fastcall sub_D4A290(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  unsigned int v7; // edx
  __int64 result; // rax

  LOWORD(v6) = sub_D4A1D0(a1, a2, a3, a4, a5, a6);
  v7 = v6;
  result = BYTE1(v6);
  if ( (_BYTE)result )
    return v7;
  return result;
}

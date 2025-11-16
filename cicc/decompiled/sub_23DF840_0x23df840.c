// Function: sub_23DF840
// Address: 0x23df840
//
__int64 __fastcall sub_23DF840(__int64 a1, unsigned __int8 a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 4) = a4;
  *(_BYTE *)(a1 + 5) = a2;
  *(_BYTE *)(a1 + 6) = a3;
  result = a3 + 32 * a2 + 2 * (unsigned int)a4;
  *(_DWORD *)a1 = result;
  return result;
}

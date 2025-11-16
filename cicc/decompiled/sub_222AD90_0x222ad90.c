// Function: sub_222AD90
// Address: 0x222ad90
//
signed __int64 __fastcall sub_222AD90(__int64 a1, char *a2, size_t a3)
{
  signed __int64 result; // rax
  int v4; // edx

  result = fread(a2, 1u, a3, *(FILE **)(a1 + 64));
  v4 = -1;
  if ( result > 0 )
    v4 = (unsigned __int8)a2[result - 1];
  *(_DWORD *)(a1 + 72) = v4;
  return result;
}

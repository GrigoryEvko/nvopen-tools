// Function: sub_AF2D80
// Address: 0xaf2d80
//
__int64 __fastcall sub_AF2D80(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rdi
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 16 - 8LL * ((v1 >> 2) & 0xF);
  result = *(_QWORD *)(v2 + 32);
  if ( result )
    return *(_QWORD *)(result + 136);
  return result;
}

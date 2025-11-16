// Function: sub_2F40590
// Address: 0x2f40590
//
bool __fastcall sub_2F40590(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, char a5)
{
  bool result; // al

  if ( (a3 & (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 920LL) + 8LL * (*(_DWORD *)(a4 + 112) & 0x7FFFFFFF)) <= 3)) == 0 )
    return *(float *)(a2 + 116) > *(float *)(a4 + 116);
  result = 1;
  if ( a5 )
    return *(float *)(a2 + 116) > *(float *)(a4 + 116);
  return result;
}

// Function: sub_16D7F90
// Address: 0x16d7f90
//
char __fastcall sub_16D7F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rax
  char result; // al

  if ( !qword_4FA1610 )
    sub_16C1EA0((__int64)&qword_4FA1610, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  v6 = qword_4FA1610;
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30C0((pthread_mutex_t **)v6);
  else
    ++*(_DWORD *)(v6 + 8);
  v7 = *(_QWORD *)(a1 + 64);
  if ( v7 )
  {
    *(_QWORD *)(v7 + 144) = a2 + 152;
    v7 = *(_QWORD *)(a1 + 64);
  }
  *(_QWORD *)(a2 + 152) = v7;
  *(_QWORD *)(a2 + 144) = a1 + 64;
  *(_QWORD *)(a1 + 64) = a2;
  result = sub_16D5D40();
  if ( result )
    return sub_16C30E0((pthread_mutex_t **)v6);
  --*(_DWORD *)(v6 + 8);
  return result;
}

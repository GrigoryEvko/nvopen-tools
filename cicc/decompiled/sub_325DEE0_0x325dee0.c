// Function: sub_325DEE0
// Address: 0x325dee0
//
char __fastcall sub_325DEE0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rsi
  char result; // al
  __int64 v5; // rdi
  __int64 v6; // rsi

  v3 = *a2;
  result = *(_QWORD *)a3 == 0 || v3 == 0;
  if ( !result )
  {
    v5 = *(_QWORD *)(v3 + 96);
    v6 = *(_QWORD *)(*(_QWORD *)a3 + 96LL);
    if ( *(_DWORD *)(v5 + 32) <= 0x40u )
      return (*(_QWORD *)(v6 + 24) & *(_QWORD *)(v5 + 24)) != 0;
    else
      return sub_C446A0((__int64 *)(v5 + 24), (__int64 *)(v6 + 24));
  }
  return result;
}

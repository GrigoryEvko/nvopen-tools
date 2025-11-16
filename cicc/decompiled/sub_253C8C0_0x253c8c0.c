// Function: sub_253C8C0
// Address: 0x253c8c0
//
__int64 *__fastcall sub_253C8C0(__int64 *a1, __int64 a2)
{
  char v2; // al
  const char *v3; // rsi

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 112LL))(a2);
  v3 = "assumed-dead";
  if ( !v2 )
    v3 = "assumed-live";
  sub_253C590(a1, v3);
  return a1;
}

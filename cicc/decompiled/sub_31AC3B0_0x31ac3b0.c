// Function: sub_31AC3B0
// Address: 0x31ac3b0
//
__int64 __fastcall sub_31AC3B0(__int64 *a1, char *a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v6; // rsi
  int v7; // eax

  v3 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu || v3 != 61 && v3 != 62 )
    return 0;
  v6 = *((_QWORD *)a2 - 4);
  if ( !v6 || !(unsigned __int8)sub_31ABA20(a1, v6, a3) )
    return 0;
  LOBYTE(v7) = sub_31A6C30((__int64)a1, *((_QWORD *)a2 + 5));
  return v7 ^ 1u;
}

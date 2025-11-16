// Function: sub_ED02A0
// Address: 0xed02a0
//
__int64 __fastcall sub_ED02A0(__int64 a1, char ***a2, _QWORD *a3)
{
  char *v3; // rax
  char *v4; // rsi

  v3 = **a2;
  v4 = &(*a2)[2][(_QWORD)(*a2)[1]];
  if ( ((unsigned __int8)v3 & 1) != 0 )
    v3 = *(char **)&v3[*(_QWORD *)v4 - 1];
  ((void (__fastcall *)(__int64, char *, _QWORD, _QWORD))v3)(a1, v4, *a3, a3[1]);
  return a1;
}

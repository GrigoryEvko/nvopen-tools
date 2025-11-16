// Function: sub_16BE230
// Address: 0x16be230
//
__int64 __fastcall sub_16BE230(__int64 a1, char *a2, __int64 a3)
{
  __int64 result; // rax

  sub_16BE1E0(a1, a2, a3);
  result = sub_16E7EE0(*(_QWORD *)(a1 + 40), a2, a3);
  *(_QWORD *)(a1 + 56) = 0;
  return result;
}

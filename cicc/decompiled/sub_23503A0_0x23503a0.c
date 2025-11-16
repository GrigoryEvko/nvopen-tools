// Function: sub_23503A0
// Address: 0x23503a0
//
_QWORD *__fastcall sub_23503A0(__int64 *a1)
{
  __int64 *v2; // rdi
  _QWORD *result; // rax
  _QWORD *v4; // rbx

  v2 = a1 + 12;
  *(v2 - 12) = 0;
  *(v2 - 11) = 0;
  *(v2 - 10) = 0;
  *((_DWORD *)v2 - 18) = 0;
  *(v2 - 8) = 0;
  *(v2 - 7) = 0;
  *(v2 - 6) = 0;
  *((_DWORD *)v2 - 10) = 0;
  *(v2 - 4) = 0;
  *(v2 - 3) = 0;
  *(v2 - 2) = 0;
  *((_DWORD *)v2 - 2) = 0;
  *v2 = 0;
  v2[1] = 0;
  v2[2] = 0;
  v2[3] = 0;
  v2[4] = 0;
  v2[5] = 0;
  v2[6] = 0;
  v2[7] = 0;
  v2[8] = 0;
  v2[9] = 0;
  sub_2350260(v2, 0);
  result = a1 + 22;
  v4 = a1 + 94;
  do
  {
    *result = 0;
    result += 4;
    *((_DWORD *)result - 2) = 0;
    *(result - 3) = 0;
    *((_DWORD *)result - 4) = 0;
    *((_DWORD *)result - 3) = 0;
  }
  while ( v4 != result );
  return result;
}

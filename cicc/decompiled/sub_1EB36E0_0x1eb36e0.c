// Function: sub_1EB36E0
// Address: 0x1eb36e0
//
_QWORD *__fastcall sub_1EB36E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v3 = a1 + 8;
  *(_QWORD *)(v3 - 8) = a2;
  sub_1EB3690(v3, 0, a2);
  sub_1EB3690(a1 + 24, 1, *(_QWORD *)a1);
  sub_1EB3690(a1 + 40, 2, *(_QWORD *)a1);
  sub_1EB3690(a1 + 56, 3, *(_QWORD *)a1);
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = a1 + 80;
  *(_QWORD *)(a1 + 104) = a1 + 80;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0x1000000000LL;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 176) = 128;
  result = (_QWORD *)sub_22077B0(6144);
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 160) = result;
  for ( i = &result[6 * *(unsigned int *)(a1 + 176)]; i != result; result += 6 )
  {
    if ( result )
    {
      result[2] = 0;
      result[3] = -8;
      *result = &unk_49FB768;
      result[1] = 2;
      result[4] = 0;
    }
  }
  *(_BYTE *)(a1 + 216) = 0;
  *(_BYTE *)(a1 + 225) = 1;
  return result;
}

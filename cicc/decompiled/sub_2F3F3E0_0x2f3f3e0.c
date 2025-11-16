// Function: sub_2F3F3E0
// Address: 0x2f3f3e0
//
_QWORD *__fastcall sub_2F3F3E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v3 = a1 + 8;
  *(_QWORD *)(v3 - 8) = a2;
  sub_2F3F390(v3, 0, a2);
  sub_2F3F390(a1 + 24, 1, *(_QWORD *)a1);
  sub_2F3F390(a1 + 40, 2, *(_QWORD *)a1);
  sub_2F3F390(a1 + 56, 3, *(_QWORD *)a1);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x600000000LL;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0x1000000000LL;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 184) = 128;
  result = (_QWORD *)sub_C7D670(6144, 8);
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 168) = result;
  for ( i = &result[6 * *(unsigned int *)(a1 + 184)]; i != result; result += 6 )
  {
    if ( result )
    {
      result[2] = 0;
      result[3] = -4096;
      *result = &unk_4A28E90;
      result[1] = 2;
      result[4] = 0;
    }
  }
  *(_BYTE *)(a1 + 224) = 0;
  return result;
}

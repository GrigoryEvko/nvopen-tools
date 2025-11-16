// Function: sub_2F3F390
// Address: 0x2f3f390
//
__int64 __fastcall sub_2F3F390(__int64 a1, int a2, __int64 a3)
{
  __int64 (*v5)(); // rdx
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  *(_QWORD *)a1 = &unk_4A2AB90;
  v5 = *(__int64 (**)())(*(_QWORD *)a3 + 168LL);
  result = 0;
  if ( v5 == sub_23CE350 )
  {
    *(_DWORD *)(a1 + 12) = 0;
  }
  else
  {
    result = ((__int64 (__fastcall *)(__int64))v5)(a3);
    *(_DWORD *)(a1 + 12) = result;
  }
  return result;
}

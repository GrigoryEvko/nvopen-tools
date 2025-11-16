// Function: sub_396F390
// Address: 0x396f390
//
unsigned __int64 __fastcall sub_396F390(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, char a5)
{
  unsigned int *v7; // r13
  unsigned __int64 result; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // rax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 282LL) && a5 )
  {
    result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 328LL))(*(_QWORD *)(a1 + 256));
    if ( a4 > 4 )
      return sub_38DD110(*(__int64 **)(a1 + 256), a4 - 4);
  }
  else
  {
    v7 = (unsigned int *)sub_38CF310(a2, 0, *(_QWORD *)(a1 + 248), 0);
    if ( a3 )
    {
      v9 = *(_QWORD *)(a1 + 248);
      v10 = sub_38CB470(a3, v9);
      v7 = (unsigned int *)sub_38CB1F0(0, (__int64)v7, v10, v9, 0);
    }
    return sub_38DDD30(*(_QWORD *)(a1 + 256), v7);
  }
  return result;
}

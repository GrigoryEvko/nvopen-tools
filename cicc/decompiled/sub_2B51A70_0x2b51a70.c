// Function: sub_2B51A70
// Address: 0x2b51a70
//
__int64 __fastcall sub_2B51A70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax

  sub_2B4DA50(*(unsigned int ***)a1, *(_QWORD *)(a1 + 8), (_QWORD *)a3, 0);
  result = sub_2B51430(
             *(_QWORD *)(a1 + 16),
             **(unsigned __int8 ****)(a1 + 8),
             *(unsigned int *)(*(_QWORD *)(a1 + 8) + 8LL),
             *(_DWORD *)(a3 + 8),
             *a2);
  *a2 = result;
  return result;
}

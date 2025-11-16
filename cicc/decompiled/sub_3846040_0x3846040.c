// Function: sub_3846040
// Address: 0x3846040
//
__int64 __fastcall sub_3846040(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax

  v4 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)a3 = *(_QWORD *)v4;
  *(_DWORD *)(a3 + 8) = *(_DWORD *)(v4 + 8);
  v5 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)a4 = *(_QWORD *)(v5 + 40);
  result = *(unsigned int *)(v5 + 48);
  *(_DWORD *)(a4 + 8) = result;
  return result;
}

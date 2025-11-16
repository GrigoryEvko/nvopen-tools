// Function: sub_1EB3690
// Address: 0x1eb3690
//
__int64 __fastcall sub_1EB3690(__int64 a1, int a2, __int64 a3)
{
  __int64 (*v5)(); // rdx
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  *(_QWORD *)a1 = &unk_49FD798;
  v5 = *(__int64 (**)())(*(_QWORD *)a3 + 544LL);
  result = 0;
  if ( v5 == sub_1EB33B0 )
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

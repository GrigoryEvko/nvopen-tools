// Function: sub_31DB460
// Address: 0x31db460
//
__int64 __fastcall sub_31DB460(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, __int64); // rbx
  __int64 v5; // rax

  v4 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 1224LL);
  v5 = sub_31DB000(a1);
  return v4(a2, a3, v5);
}

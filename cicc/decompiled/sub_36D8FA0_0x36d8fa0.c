// Function: sub_36D8FA0
// Address: 0x36d8fa0
//
bool __fastcall sub_36D8FA0(__int64 a1)
{
  __int64 v1; // rdi
  __int64 (__fastcall *v2)(__int64); // rax
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 1136);
  v2 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 144LL);
  if ( v2 == sub_3020010 )
    return sub_3037D40(v1 + 960);
  v4 = ((__int64 (*)(void))v2)();
  return sub_3037D40(v4);
}

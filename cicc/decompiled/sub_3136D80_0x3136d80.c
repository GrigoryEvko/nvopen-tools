// Function: sub_3136D80
// Address: 0x3136d80
//
void __fastcall sub_3136D80(__int64 a1, __int64 a2)
{
  __int64 i; // r12

  for ( i = *(_QWORD *)(a1 + 88); a1 + 72 != i; i = sub_220EEE0(i) )
    (*(void (__fastcall **)(_QWORD, __int64, __int64))a2)(*(_QWORD *)(a2 + 8), i + 32, i + 80);
}

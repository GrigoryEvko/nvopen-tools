// Function: sub_39B0CD0
// Address: 0x39b0cd0
//
__int64 __fastcall sub_39B0CD0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 (*v5)(); // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rdx
  __int64 result; // rax

  *a1 = sub_1632FA0(*(_QWORD *)(a3 + 40));
  v5 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
  if ( v5 == sub_16FF750 )
  {
    a1[1] = 0;
    BUG();
  }
  v6 = ((__int64 (__fastcall *)(__int64, __int64))v5)(a2, a3);
  a1[1] = v6;
  v7 = v6;
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 56LL);
  result = 0;
  if ( v8 == sub_1D12D20 )
  {
    a1[2] = 0;
  }
  else
  {
    result = ((__int64 (__fastcall *)(__int64))v8)(v7);
    a1[2] = result;
  }
  return result;
}

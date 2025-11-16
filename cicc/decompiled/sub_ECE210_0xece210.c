// Function: sub_ECE210
// Address: 0xece210
//
__int64 __fastcall sub_ECE210(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax

  if ( a2 == 9 )
    return sub_ECE1B0(a1, a3);
  if ( a2 == *(_DWORD *)sub_ECD7B0(a1) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    return 0;
  }
  else
  {
    v5 = sub_ECD7B0(a1);
    v6 = sub_ECD6A0(v5);
    return sub_ECDA70(a1, v6, a3, 0, 0);
  }
}

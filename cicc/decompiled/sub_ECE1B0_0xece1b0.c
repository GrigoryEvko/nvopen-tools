// Function: sub_ECE1B0
// Address: 0xece1b0
//
__int64 __fastcall sub_ECE1B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax

  if ( *(_DWORD *)sub_ECD7B0(a1) == 9 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    return 0;
  }
  else
  {
    v3 = sub_ECD7B0(a1);
    v4 = sub_ECD6A0(v3);
    return sub_ECDA70(a1, v4, a2, 0, 0);
  }
}

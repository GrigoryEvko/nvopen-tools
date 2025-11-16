// Function: sub_EC51B0
// Address: 0xec51b0
//
__int64 __fastcall sub_EC51B0(__int64 a1)
{
  __int64 v2; // rdi
  const char *v3; // [rsp+0h] [rbp-40h] BYREF
  char v4; // [rsp+20h] [rbp-20h]
  char v5; // [rsp+21h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 1520) = 0;
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 8);
    v5 = 1;
    v3 = "unexpected token in '.secure_log_reset' directive";
    v4 = 3;
    return sub_ECE0E0(v2, &v3, 0, 0);
  }
}

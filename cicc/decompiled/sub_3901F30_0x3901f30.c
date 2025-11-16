// Function: sub_3901F30
// Address: 0x3901f30
//
__int64 __fastcall sub_3901F30(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 v4; // rdi
  const char *v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]
  char v7; // [rsp+11h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 744) = 0;
    return 0;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 8);
    v7 = 1;
    v5 = "unexpected token in '.secure_log_reset' directive";
    v6 = 3;
    return sub_3909CF0(v4, &v5, 0, 0, v1, v2);
  }
}

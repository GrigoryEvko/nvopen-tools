// Function: sub_3902030
// Address: 0x3902030
//
__int64 __fastcall sub_3902030(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 v3; // rdi
  void (*v4)(); // rdx
  __int64 result; // rax
  __int64 v6; // rdi
  const char *v7; // [rsp+10h] [rbp-30h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v4 = *(void (**)())(*(_QWORD *)v3 + 208LL);
    result = 0;
    if ( v4 != nullsub_583 )
    {
      ((void (__fastcall *)(__int64, __int64))v4)(v3, 4);
      return 0;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 8);
    v9 = 1;
    v7 = "unexpected token in '.end_data_region' directive";
    v8 = 3;
    return sub_3909CF0(v6, &v7, 0, 0, v1, v2);
  }
  return result;
}

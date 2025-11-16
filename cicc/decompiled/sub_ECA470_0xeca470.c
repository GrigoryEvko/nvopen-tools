// Function: sub_ECA470
// Address: 0xeca470
//
__int64 __fastcall sub_ECA470(__int64 a1)
{
  const char *v1; // rax
  __int64 v2; // rdi
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdi
  void (*v9)(); // rax
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v12 = 1;
    v1 = "expected string";
LABEL_3:
    v2 = *(_QWORD *)(a1 + 8);
    v10 = v1;
    v11 = 3;
    return sub_ECE0E0(v2, &v10, 0, 0);
  }
  v4 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v4 == 2 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    v5 = *(_QWORD *)(v4 + 16);
  }
  else
  {
    v5 = *(_QWORD *)(v4 + 16);
    v6 = *(_QWORD *)(v4 + 8);
    if ( v5 )
    {
      v7 = v5 - 1;
      if ( !v7 )
        v7 = 1;
      ++v6;
      v5 = v7 - 1;
    }
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    v12 = 1;
    v1 = "expected end of directive";
    goto LABEL_3;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v9 = *(void (**)())(*(_QWORD *)v8 + 648LL);
  if ( v9 != nullsub_107 )
    ((void (__fastcall *)(__int64, __int64, __int64))v9)(v8, v6, v5);
  return 0;
}

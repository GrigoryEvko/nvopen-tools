// Function: sub_ECE770
// Address: 0xece770
//
__int64 __fastcall sub_ECE770(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdi
  void (*v8)(); // rax
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
    goto LABEL_2;
  v3 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v3 == 2 )
  {
    v5 = *(_QWORD *)(v3 + 8);
    v4 = *(_QWORD *)(v3 + 16);
  }
  else
  {
    v4 = *(_QWORD *)(v3 + 16);
    v5 = *(_QWORD *)(v3 + 8);
    if ( v4 )
    {
      v6 = v4 - 1;
      if ( !v6 )
        v6 = 1;
      ++v5;
      v4 = v6 - 1;
    }
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v8 = *(void (**)())(*(_QWORD *)v7 + 648LL);
    if ( v8 != nullsub_107 )
      ((void (__fastcall *)(__int64, __int64, __int64))v8)(v7, v5, v4);
    return 0;
  }
  else
  {
LABEL_2:
    v1 = *(_QWORD *)(a1 + 8);
    v11 = 1;
    v9 = "unexpected token in '.ident' directive";
    v10 = 3;
    return sub_ECE0E0(v1, (__int64)&v9, 0, 0);
  }
}

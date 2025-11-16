// Function: sub_3906740
// Address: 0x3906740
//
__int64 __fastcall sub_3906740(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 v3; // rdi
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r12
  __int64 v10; // rdi
  void (*v11)(); // rax
  const char *v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]
  char v14; // [rsp+11h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
    goto LABEL_2;
  v5 = sub_3909460(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v5 == 2 )
  {
    v9 = *(_QWORD *)(v5 + 8);
    v7 = *(_QWORD *)(v5 + 16);
  }
  else
  {
    v6 = *(_QWORD *)(v5 + 16);
    v7 = 0;
    if ( v6 )
    {
      v8 = v6 - 1;
      if ( v6 == 1 )
        v8 = 1;
      if ( v8 > v6 )
        v8 = *(_QWORD *)(v5 + 16);
      v6 = 1;
      v7 = v8 - 1;
    }
    v9 = *(_QWORD *)(v5 + 8) + v6;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v11 = *(void (**)())(*(_QWORD *)v10 + 560LL);
    if ( v11 != nullsub_589 )
      ((void (__fastcall *)(__int64, unsigned __int64, unsigned __int64))v11)(v10, v9, v7);
    return 0;
  }
  else
  {
LABEL_2:
    v3 = *(_QWORD *)(a1 + 8);
    v14 = 1;
    v12 = "unexpected token in '.ident' directive";
    v13 = 3;
    return sub_3909CF0(v3, &v12, 0, 0, v1, v2);
  }
}

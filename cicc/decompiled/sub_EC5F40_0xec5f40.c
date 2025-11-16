// Function: sub_EC5F40
// Address: 0xec5f40
//
__int64 __fastcall sub_EC5F40(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned int v5; // r12d
  __int64 v6; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  void (*v10)(); // rax
  unsigned int v11; // r13d
  __int64 v12; // rdi
  void (*v13)(); // rax
  __int64 v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[4]; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-30h]
  char v18; // [rsp+31h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v10 = *(void (**)())(*(_QWORD *)v9 + 240LL);
    if ( v10 != nullsub_101 )
      ((void (__fastcall *)(__int64, _QWORD))v10)(v9, 0);
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 8);
    v14 = 0;
    v15 = 0;
    v3 = sub_ECD7B0(v2);
    v4 = sub_ECD6A0(v3);
    v5 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &v14);
    if ( (_BYTE)v5 )
    {
      v6 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      v16[0] = "expected region type after '.data_region' directive";
      v17 = 3;
      return (unsigned int)sub_ECE0E0(v6, v16, 0, 0);
    }
    if ( v15 == 3 )
    {
      if ( *(_WORD *)v14 != 29802 || *(_BYTE *)(v14 + 2) != 56 )
        goto LABEL_9;
      v11 = 1;
    }
    else
    {
      if ( v15 != 4 )
      {
LABEL_9:
        v8 = *(_QWORD *)(a1 + 8);
        v18 = 1;
        v16[0] = "unknown region type in '.data_region' directive";
        v17 = 3;
        return (unsigned int)sub_ECDA70(v8, v4, v16, 0, 0);
      }
      if ( *(_DWORD *)v14 == 909210730 )
      {
        v11 = 2;
      }
      else
      {
        if ( *(_DWORD *)v14 != 842232938 )
          goto LABEL_9;
        v11 = 3;
      }
    }
    (*(void (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8), &v14, v14);
    v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v13 = *(void (**)())(*(_QWORD *)v12 + 240LL);
    if ( v13 != nullsub_101 )
    {
      ((void (__fastcall *)(__int64, _QWORD))v13)(v12, v11);
      return v5;
    }
  }
  return 0;
}

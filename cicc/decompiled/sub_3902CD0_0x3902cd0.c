// Function: sub_3902CD0
// Address: 0x3902cd0
//
__int64 __fastcall sub_3902CD0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // r12d
  __int64 v8; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  void (*v12)(); // rax
  unsigned int v13; // r13d
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // [rsp+0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+8h] [rbp-48h]
  _QWORD v18[2]; // [rsp+10h] [rbp-40h] BYREF
  char v19; // [rsp+20h] [rbp-30h]
  char v20; // [rsp+21h] [rbp-2Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v12 = *(void (**)())(*(_QWORD *)v11 + 208LL);
    if ( v12 != nullsub_583 )
      ((void (__fastcall *)(__int64, _QWORD))v12)(v11, 0);
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 8);
    v16 = 0;
    v17 = 0;
    v3 = sub_3909460(v2);
    v4 = sub_39092A0(v3);
    v7 = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 144LL))(*(_QWORD *)(a1 + 8), &v16);
    if ( (_BYTE)v7 )
    {
      v10 = *(_QWORD *)(a1 + 8);
      v20 = 1;
      v18[0] = "expected region type after '.data_region' directive";
      v19 = 3;
      return (unsigned int)sub_3909CF0(v10, v18, 0, 0, v5, v6);
    }
    if ( v17 == 3 )
    {
      if ( *(_WORD *)v16 != 29802 || *(_BYTE *)(v16 + 2) != 56 )
        goto LABEL_5;
      v13 = 1;
    }
    else
    {
      if ( v17 != 4 )
      {
LABEL_5:
        v8 = *(_QWORD *)(a1 + 8);
        v20 = 1;
        v18[0] = "unknown region type in '.data_region' directive";
        v19 = 3;
        return (unsigned int)sub_3909790(v8, v4, v18, 0, 0);
      }
      if ( *(_DWORD *)v16 == 909210730 )
      {
        v13 = 2;
      }
      else
      {
        if ( *(_DWORD *)v16 != 842232938 )
          goto LABEL_5;
        v13 = 3;
      }
    }
    (*(void (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8), &v16, v16);
    v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v15 = *(void (**)())(*(_QWORD *)v14 + 208LL);
    if ( v15 != nullsub_583 )
    {
      ((void (__fastcall *)(__int64, _QWORD))v15)(v14, v13);
      return v7;
    }
  }
  return 0;
}

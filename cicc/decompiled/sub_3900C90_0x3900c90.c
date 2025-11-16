// Function: sub_3900C90
// Address: 0x3900c90
//
__int64 __fastcall sub_3900C90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  const char *v11; // rax
  __int64 v12; // rdi
  _QWORD v14[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]

  v3 = 0;
  if ( a3 == 5 )
  {
    if ( *(_DWORD *)a2 != 1634039598 || (v3 = 20, *(_BYTE *)(a2 + 4) != 107) )
      v3 = 0;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
LABEL_10:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    return 0;
  }
  else
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(a1 + 8);
      v14[0] = 0;
      v14[1] = 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v10 + 144LL))(v10, v14) )
      {
        HIBYTE(v16) = 1;
        v11 = "expected identifier in directive";
        goto LABEL_9;
      }
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v15[0] = v14;
      v16 = 261;
      v6 = sub_38BF510(v5, (__int64)v15);
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 256LL))(v7, v6, v3);
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
        goto LABEL_10;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
        break;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    }
    HIBYTE(v16) = 1;
    v11 = "unexpected token in directive";
LABEL_9:
    v12 = *(_QWORD *)(a1 + 8);
    v15[0] = v11;
    LOBYTE(v16) = 3;
    return sub_3909CF0(v12, v15, 0, 0, v8, v9);
  }
}

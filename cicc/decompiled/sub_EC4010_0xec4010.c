// Function: sub_EC4010
// Address: 0xec4010
//
__int64 __fastcall sub_EC4010(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdi
  const char *v9; // rax
  __int64 v10; // rdi
  const char *v12; // [rsp+0h] [rbp-70h] BYREF
  const char *v13; // [rsp+8h] [rbp-68h]
  const char *v14[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  if ( a3 != 5 )
  {
    if ( a3 == 14
      && *(_QWORD *)a2 == 0x6E615F6B6165772ELL
      && *(_DWORD *)(a2 + 8) == 1683974516
      && *(_WORD *)(a2 + 12) == 28773 )
    {
      v4 = 28;
      goto LABEL_4;
    }
LABEL_3:
    v4 = 0;
    goto LABEL_4;
  }
  if ( *(_DWORD *)a2 != 1634039598 )
    goto LABEL_3;
  v4 = 24;
  if ( *(_BYTE *)(a2 + 4) != 107 )
    goto LABEL_3;
LABEL_4:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
LABEL_17:
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    return 0;
  }
  else
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v12 = 0;
      v13 = 0;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v8 + 192LL))(v8, &v12) )
      {
        HIBYTE(v15) = 1;
        v9 = "expected identifier in directive";
        goto LABEL_11;
      }
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v15 = 261;
      v14[0] = v12;
      v14[1] = v13;
      v6 = sub_E6C460(v5, v14);
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v7 + 296LL))(v7, v6, v4);
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
        goto LABEL_17;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
        break;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    }
    HIBYTE(v15) = 1;
    v9 = "unexpected token in directive";
LABEL_11:
    v10 = *(_QWORD *)(a1 + 8);
    v14[0] = v9;
    LOBYTE(v15) = 3;
    return sub_ECE0E0(v10, v14, 0, 0);
  }
}

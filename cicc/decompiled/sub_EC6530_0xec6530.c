// Function: sub_EC6530
// Address: 0xec6530
//
__int64 __fastcall sub_EC6530(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v13; // rdi
  __int64 v14; // rdi
  const char *v15; // rax
  const char *v16; // [rsp+0h] [rbp-60h] BYREF
  const char *v17; // [rsp+8h] [rbp-58h]
  const char *v18[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v19; // [rsp+30h] [rbp-30h]

  v6 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8))
                                                  + 288)
                                      + 8LL)
                          + 164LL);
  if ( (unsigned int)(v6 - 6) > 2 && v6 != 20 )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v18[0] = "indirect symbol not in a symbol pointer or stub section";
    v19 = 259;
    return (unsigned int)sub_ECDA70(v13, a4, v18, 0, 0);
  }
  v7 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v17 = 0;
  v8 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v7 + 192LL))(v7, &v16);
  if ( (_BYTE)v8 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v18[0] = "expected identifier in .indirect_symbol directive";
    v19 = 259;
    return (unsigned int)sub_ECE0E0(v14, v18, 0, 0);
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v19 = 261;
  v18[0] = v16;
  v18[1] = v17;
  v10 = sub_E6C460(v9, v18);
  if ( (*(_BYTE *)(v10 + 8) & 2) != 0 )
  {
    HIBYTE(v19) = 1;
    v15 = "non-local symbol required in directive";
  }
  else
  {
    v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11 + 296LL))(v11, v10, 14) )
    {
      v19 = 1283;
      v18[0] = "unable to emit indirect symbol attribute for: ";
      v18[2] = v16;
      v18[3] = v17;
      return (unsigned int)sub_ECE0E0(*(_QWORD *)(a1 + 8), v18, 0, 0);
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      return v8;
    }
    HIBYTE(v19) = 1;
    v15 = "unexpected token in '.indirect_symbol' directive";
  }
  v18[0] = v15;
  LOBYTE(v19) = 3;
  return (unsigned int)sub_ECE0E0(*(_QWORD *)(a1 + 8), v18, 0, 0);
}

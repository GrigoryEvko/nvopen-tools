// Function: sub_E5EC20
// Address: 0xe5ec20
//
__int64 __fastcall sub_E5EC20(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v6; // r14
  unsigned int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned __int8 v13; // [rsp+7h] [rbp-59h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h] BYREF
  const char *v15; // [rsp+10h] [rbp-50h] BYREF
  char v16; // [rsp+30h] [rbp-30h]
  char v17; // [rsp+31h] [rbp-2Fh]

  v4 = a1[1];
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 160LL);
  if ( v5 != sub_E5B860
    && ((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64, unsigned __int8 *))v5)(v4, a1, a2, &v13) )
  {
    return v13;
  }
  v6 = *a1;
  v7 = sub_E81920(*(_QWORD *)(a2 + 112), &v14);
  if ( (_BYTE)v7 )
  {
    v8 = *(_QWORD *)(a2 + 48);
    v9 = v14;
    *(_QWORD *)(a2 + 48) = 0;
    *(_DWORD *)(a2 + 80) = 0;
    sub_E78020(v6, v9, a2 + 40);
    LOBYTE(v7) = *(_QWORD *)(a2 + 48) != v8;
  }
  else
  {
    v17 = 1;
    v11 = *a1;
    v15 = "invalid CFI advance_loc expression";
    v12 = *(_QWORD *)(a2 + 112);
    v16 = 3;
    sub_E66880(v11, *(_QWORD *)(v12 + 8), &v15);
    *(_QWORD *)(a2 + 112) = sub_E81A90(0, v6, 0, 0);
  }
  return v7;
}

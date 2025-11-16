// Function: sub_B6EB20
// Address: 0xb6eb20
//
_BYTE *__fastcall sub_B6EB20(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE *result; // rax
  __int64 v7; // r12
  char *v8; // rax
  char *v9; // r13
  size_t v10; // rax
  _WORD *v11; // rdi
  size_t v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  _QWORD v16[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( (unsigned int)(*(_DWORD *)(a2 + 8) - 13) <= 8 )
  {
    v3 = sub_B6EA50(a1);
    if ( v3 )
      sub_B7CDF0(v3, a2);
  }
  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 104LL);
  if ( !v5 )
    goto LABEL_8;
  if ( *(_BYTE *)(a2 + 12) )
  {
    if ( !*(_BYTE *)(v4 + 112) )
      goto LABEL_7;
LABEL_11:
    if ( !(unsigned __int8)sub_B6E560(a2) )
      goto LABEL_8;
    v4 = *(_QWORD *)a1;
    goto LABEL_7;
  }
  *(_BYTE *)(v5 + 16) = 1;
  v4 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 112LL) )
    goto LABEL_11;
LABEL_7:
  result = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v4 + 104) + 16LL))(
                      *(_QWORD *)(v4 + 104),
                      a2);
  if ( (_BYTE)result )
    return result;
LABEL_8:
  result = (_BYTE *)sub_B6E560(a2);
  if ( !(_BYTE)result )
    return result;
  v16[1] = ((__int64 (*)(void))sub_CB72A0)();
  v16[0] = &unk_49E1428;
  v7 = sub_CB72A0(a2, a2);
  v8 = sub_B6EAD0(*(_BYTE *)(a2 + 12));
  v9 = v8;
  if ( !v8 )
    goto LABEL_18;
  v10 = strlen(v8);
  v11 = *(_WORD **)(v7 + 32);
  v12 = v10;
  v13 = *(_QWORD *)(v7 + 24) - (_QWORD)v11;
  if ( v12 > v13 )
  {
    v7 = sub_CB6200(v7, v9, v12);
LABEL_18:
    v11 = *(_WORD **)(v7 + 32);
    v13 = *(_QWORD *)(v7 + 24) - (_QWORD)v11;
    goto LABEL_19;
  }
  if ( v12 )
  {
    memcpy(v11, v9, v12);
    v14 = *(_QWORD *)(v7 + 24);
    v11 = (_WORD *)(v12 + *(_QWORD *)(v7 + 32));
    *(_QWORD *)(v7 + 32) = v11;
    v13 = v14 - (_QWORD)v11;
  }
LABEL_19:
  if ( v13 <= 1 )
  {
    sub_CB6200(v7, ": ", 2);
  }
  else
  {
    *v11 = 8250;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 24LL))(a2, v16);
  v15 = sub_CB72A0(a2, v16);
  result = *(_BYTE **)(v15 + 32);
  if ( *(_BYTE **)(v15 + 24) == result )
  {
    result = (_BYTE *)sub_CB6200(v15, "\n", 1);
    if ( !*(_BYTE *)(a2 + 12) )
LABEL_23:
      exit(1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v15 + 32);
    if ( !*(_BYTE *)(a2 + 12) )
      goto LABEL_23;
  }
  return result;
}

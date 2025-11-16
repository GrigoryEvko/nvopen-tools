// Function: sub_2E84FA0
// Address: 0x2e84fa0
//
__int64 __fastcall sub_2E84FA0(__int64 a1, __int64 *a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rdi
  _WORD *v7; // rdx
  __int64 v8; // rax
  _WORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx

  v3 = (char *)sub_2E791E0(a2);
  if ( !sub_BC63A0(v3, v4) )
    return 0;
  v6 = *(_QWORD *)(a1 + 200);
  v7 = *(_WORD **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 1u )
  {
    v6 = sub_CB6200(v6, (unsigned __int8 *)"# ", 2u);
  }
  else
  {
    *v7 = 8227;
    *(_QWORD *)(v6 + 32) += 2LL;
  }
  v8 = sub_CB6200(v6, *(unsigned __int8 **)(a1 + 208), *(_QWORD *)(a1 + 216));
  v9 = *(_WORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 1u )
  {
    sub_CB6200(v8, (unsigned __int8 *)":\n", 2u);
  }
  else
  {
    *v9 = 2618;
    *(_QWORD *)(v8 + 32) += 2LL;
  }
  v10 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5025C1C);
  if ( v10 && (v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_5025C1C)) != 0 )
    v12 = v11 + 200;
  else
    v12 = 0;
  sub_2E823F0((__int64)a2, *(_QWORD *)(a1 + 200), v12);
  return 0;
}

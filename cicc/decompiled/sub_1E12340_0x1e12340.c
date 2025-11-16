// Function: sub_1E12340
// Address: 0x1e12340
//
__int64 __fastcall sub_1E12340(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdi
  _WORD *v5; // rdx
  __int64 v6; // rax
  _WORD *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx

  sub_1E0A440(a2);
  if ( !(unsigned __int8)sub_160E740() )
    return 0;
  v4 = *(_QWORD *)(a1 + 232);
  v5 = *(_WORD **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 1u )
  {
    v4 = sub_16E7EE0(v4, (char *)"# ", 2u);
  }
  else
  {
    *v5 = 8227;
    *(_QWORD *)(v4 + 24) += 2LL;
  }
  v6 = sub_16E7EE0(v4, *(char **)(a1 + 240), *(_QWORD *)(a1 + 248));
  v7 = *(_WORD **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 1u )
  {
    sub_16E7EE0(v6, ":\n", 2u);
  }
  else
  {
    *v7 = 2618;
    *(_QWORD *)(v6 + 24) += 2LL;
  }
  v8 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCA82C, 1u);
  v9 = v8;
  if ( v8 )
    v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4FCA82C);
  sub_1E0B0B0((__int64)a2, *(_QWORD *)(a1 + 232), v9);
  return 0;
}

// Function: sub_18FBB40
// Address: 0x18fbb40
//
char __fastcall sub_18FBB40(__int64 a1, int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v8; // r13
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx

  if ( a2 == a3 )
    return 1;
  v6 = *(_QWORD *)(a1 + 72);
  if ( !v6 )
    return 0;
  v8 = sub_1422850(v6, a4);
  if ( !v8 || !sub_1422850(*(_QWORD *)(a1 + 72), a5) )
    return 1;
  v9 = (__int64 *)sub_1423BA0(*(__int64 **)(a1 + 72));
  v10 = sub_1422850(v9[1], a5);
  v11 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*v9 + 16))(v9, v10);
  return sub_1428550(*(_QWORD *)(a1 + 72), v11, v8, v12);
}

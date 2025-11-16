// Function: sub_7DC550
// Address: 0x7dc550
//
__int64 __fastcall sub_7DC550(unsigned int a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r12
  _BYTE *v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // r14
  _QWORD *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // r14
  _BYTE *v16; // rax
  _BYTE *v17; // r12
  _QWORD *v18; // rax
  _QWORD *v20; // rax

  v5 = sub_7DC1A0();
  v6 = sub_7E7CA0(v5);
  *a2 = v6;
  v7 = qword_4F188A8;
  v8 = v6;
  v9 = sub_731250(v6);
  v10 = sub_73DE50((__int64)v9, v7);
  v11 = qword_4F188D8;
  v12 = v10;
  if ( !qword_4F188D8 )
  {
    v20 = (_QWORD *)sub_7DC1A0();
    sub_72D2E0(v20);
    qword_4F188D8 = sub_7E2190("__curr_eh_stack_entry");
    v11 = qword_4F188D8;
  }
  v13 = sub_73E830(v11);
  sub_7E6A80(v12, 73, v13, a3);
  v14 = sub_73E230(v8, 73);
  sub_7E6AB0(qword_4F188D8, v14, a3);
  v15 = qword_4F188A0;
  v16 = sub_731250(v8);
  v17 = sub_73DE50((__int64)v16, v15);
  v18 = sub_73A830(a1, 2u);
  return sub_7E6A80(v17, 73, v18, a3);
}

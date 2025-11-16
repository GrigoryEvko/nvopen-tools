// Function: sub_7DC1A0
// Address: 0x7dc1a0
//
__int64 sub_7DC1A0()
{
  __int64 result; // rax
  __int64 v1; // r15
  __int64 v2; // r14
  _QWORD *v3; // rax
  unsigned __int64 v4; // rbx
  _QWORD *v5; // rsi
  __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // rdi

  result = qword_4F188B0;
  if ( !qword_4F188B0 )
  {
    qword_4F188B0 = sub_7E16B0(10);
    sub_7E1CA0(qword_4F188B0);
    v1 = sub_7E16B0(11);
    sub_7E1CA0(v1);
    v2 = sub_7E16B0(10);
    sub_7E1CA0(v2);
    if ( !qword_4F188B8 )
    {
      qword_4F188B8 = (__int64)sub_7259C0(8);
      if ( unk_4F0695C )
        v13 = sub_72C610(unk_4F06958);
      else
        v13 = sub_72BA30(unk_4F06959);
      v14 = qword_4F188B8;
      *(_QWORD *)(qword_4F188B8 + 160) = v13;
      *(_QWORD *)(v14 + 176) = unk_4F06960;
      sub_8D6090(v14);
    }
    sub_7E1B70("setjmp_buffer");
    qword_4F18888 = 0;
    v3 = (_QWORD *)sub_7DC0C0();
    v4 = sub_72D2E0(v3);
    sub_7E1B70("catch_entries");
    qword_4F18880 = 0;
    sub_7E1C10("catch_entries", v4);
    sub_7E1B70("rtinfo");
    qword_4F18878 = 0;
    v5 = sub_72BA30(unk_4F06871);
    sub_7E1B70((char *)"region_number");
    qword_4F18870 = 0;
    sub_7E1C00(v2, (unsigned __int64)v5);
    v6 = sub_7E16B0(10);
    sub_7E1CA0(v6);
    v7 = (_QWORD *)sub_7DB5D0();
    v8 = sub_72D2E0(v7);
    sub_7E1B70("regions");
    qword_4F18860 = 0;
    v9 = (_QWORD *)sub_7E1C10("regions", v8);
    v10 = sub_72D2E0(v9);
    sub_7E1B70("obj_table");
    qword_4F18858 = 0;
    v11 = (_QWORD *)sub_7DAF40("obj_table", v10);
    sub_72D2E0(v11);
    sub_7E1B70("array_table");
    qword_4F18850 = 0;
    v12 = sub_72BA30(unk_4F06871);
    sub_7E1B70("saved_region_number");
    qword_4F18848 = 0;
    sub_7E1C00(v6, (unsigned __int64)v12);
    sub_7E1B70("try_block");
    qword_4F18890 = 0;
    sub_7E1B70("function");
    qword_4F18868 = 0;
    sub_7E1B70("throw_spec");
    qword_4F18840 = 0;
    sub_7E1C00(v1, v4);
    sub_72D2E0((_QWORD *)qword_4F188B0);
    sub_7E1B70("next");
    qword_4F188A8 = 0;
    sub_72BA30(2u);
    sub_7E1B70("kind");
    qword_4F188A0 = 0;
    sub_7E1B70("variant");
    qword_4F18898 = 0;
    sub_7E1C00(qword_4F188B0, v1);
    return qword_4F188B0;
  }
  return result;
}

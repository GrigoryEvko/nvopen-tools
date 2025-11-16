// Function: sub_33650A0
// Address: 0x33650a0
//
__int64 __fastcall sub_33650A0(int a1, __int64 a2, __int64 a3, int a4)
{
  __int128 v6; // rax
  int v7; // r9d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // r14
  __int128 v12; // rax
  int v13; // r9d
  __int128 v14; // rax
  int v15; // r9d
  __int128 v17; // [rsp-28h] [rbp-50h]
  __int128 v18; // [rsp-28h] [rbp-50h]

  *(_QWORD *)&v6 = sub_3400BD0(a1, 0x7FFFFF, a4, 7, 0, 0, 0);
  *((_QWORD *)&v17 + 1) = a3;
  *(_QWORD *)&v17 = a2;
  v8 = sub_3406EB0(a1, 186, a4, 7, 0, v7, v17, v6);
  v10 = v9;
  v11 = v8;
  *(_QWORD *)&v12 = sub_3400BD0(a1, 1065353216, a4, 7, 0, 0, 0);
  *((_QWORD *)&v18 + 1) = v10;
  *(_QWORD *)&v18 = v11;
  *(_QWORD *)&v14 = sub_3406EB0(a1, 187, a4, 7, 0, v13, v18, v12);
  return sub_33FAF80(a1, 234, a4, 12, 0, v15, v14);
}

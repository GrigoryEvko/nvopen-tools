// Function: sub_3365170
// Address: 0x3365170
//
__int64 __fastcall sub_3365170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int128 v8; // rax
  int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  int v15; // eax
  int v16; // edx
  __int128 v17; // rax
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r14
  __int128 v23; // rax
  int v24; // r9d
  __int128 v25; // rax
  int v26; // r9d
  __int128 v28; // [rsp-28h] [rbp-60h]
  __int128 v29; // [rsp-28h] [rbp-60h]
  __int128 v30; // [rsp-28h] [rbp-60h]

  *(_QWORD *)&v8 = sub_3400BD0(a1, 2139095040, a5, 7, 0, 0, 0);
  *((_QWORD *)&v28 + 1) = a3;
  *(_QWORD *)&v28 = a2;
  v10 = sub_3406EB0(a1, 186, a5, 7, 0, v9, v28, v8);
  v12 = v11;
  v13 = v10;
  v14 = sub_2E79000(*(__int64 **)(a1 + 40));
  v15 = sub_2FE6750(a4, 7, 0, v14);
  *(_QWORD *)&v17 = sub_3400BD0(a1, 23, a5, v15, v16, 0, 0);
  *((_QWORD *)&v29 + 1) = v12;
  *(_QWORD *)&v29 = v13;
  v19 = sub_3406EB0(a1, 192, a5, 7, 0, v18, v29, v17);
  v21 = v20;
  v22 = v19;
  *(_QWORD *)&v23 = sub_3400BD0(a1, 127, a5, 7, 0, 0, 0);
  *((_QWORD *)&v30 + 1) = v21;
  *(_QWORD *)&v30 = v22;
  *(_QWORD *)&v25 = sub_3406EB0(a1, 57, a5, 7, 0, v24, v30, v23);
  return sub_33FAF80(a1, 220, a5, 12, 0, v26, v25);
}

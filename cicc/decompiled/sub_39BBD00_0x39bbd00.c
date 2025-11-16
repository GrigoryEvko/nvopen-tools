// Function: sub_39BBD00
// Address: 0x39bbd00
//
void __fastcall sub_39BBD00(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  char *v10; // rax
  __int64 v11; // r9
  char *v12; // r10
  char *v13; // r11
  __int64 v14; // r15
  char *v15; // r13
  char *v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // [rsp-58h] [rbp-58h]
  __int64 v21; // [rsp-50h] [rbp-50h]
  char *v22; // [rsp-50h] [rbp-50h]
  __int64 *v23; // [rsp-50h] [rbp-50h]
  char *v24; // [rsp-48h] [rbp-48h]
  char *v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]
  __int64 v28; // [rsp-30h] [rbp-30h]
  __int64 v29; // [rsp-20h] [rbp-20h]
  __int64 v30; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v30 = v7;
    v29 = v6;
    v8 = a5;
    v28 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v17 = *(_QWORD *)a1;
      v18 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)a2 + 16LL))(*(_QWORD *)a2);
      if ( v18 < (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 16LL))(v17) )
      {
        v19 = *(_QWORD *)a1;
        *(_QWORD *)a1 = *(_QWORD *)a2;
        *(_QWORD *)a2 = v19;
      }
      return;
    }
    if ( a4 > a5 )
    {
      v26 = a3;
      v14 = a4 / 2;
      v23 = (__int64 *)&a1[8 * (a4 / 2)];
      v16 = (char *)sub_39BB170(a2, a3, v23);
      v11 = v26;
      v13 = (char *)v23;
      v12 = v16;
      v27 = (v16 - a2) >> 3;
    }
    else
    {
      v21 = a3;
      v27 = a5 / 2;
      v24 = &a2[8 * (a5 / 2)];
      v10 = (char *)sub_39BB0E0(a1, (__int64)a2, v24);
      v11 = v21;
      v12 = v24;
      v13 = v10;
      v14 = (v10 - a1) >> 3;
    }
    v20 = v11;
    v22 = v12;
    v25 = v13;
    v15 = sub_39BBB40(v13, a2, v12);
    sub_39BBD00(a1, v25, v15, v14, v27);
    a4 = v9 - v14;
    a1 = v15;
    v5 = v28;
    a5 = v8 - v27;
    a3 = v20;
    a2 = v22;
    v6 = v29;
    v7 = v30;
  }
}

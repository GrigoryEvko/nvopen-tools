// Function: sub_1890E40
// Address: 0x1890e40
//
void __fastcall sub_1890E40(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r15
  char *v11; // r9
  char *v12; // r10
  char *v13; // r11
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  char *v18; // [rsp-58h] [rbp-58h]
  char *v19; // [rsp-50h] [rbp-50h]
  char *v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-30h] [rbp-30h]
  __int64 v23; // [rsp-18h] [rbp-18h]
  __int64 v24; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v24 = v7;
    v23 = v6;
    v8 = a5;
    v22 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v15 = *(_QWORD *)a1;
      if ( *(_QWORD *)a2 < *(_QWORD *)a1 )
      {
        *(_QWORD *)a1 = *(_QWORD *)a2;
        v16 = *((_QWORD *)a2 + 1);
        *(_QWORD *)a2 = v15;
        v17 = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = v16;
        *((_QWORD *)a2 + 1) = v17;
      }
      return;
    }
    v10 = a3;
    if ( a4 > a5 )
    {
      v14 = a4 / 2;
      v12 = (char *)sub_18906F0(a2, a3, (unsigned __int64 *)&a1[16 * (a4 / 2)]);
      v21 = (v12 - v11) >> 4;
    }
    else
    {
      v21 = a5 / 2;
      v13 = (char *)sub_1890750(a1, (__int64)a2, (unsigned __int64 *)&a2[16 * (a5 / 2)]);
      v14 = (v13 - a1) >> 4;
    }
    v18 = v12;
    v19 = v13;
    v20 = sub_18904C0(v13, v11, v12);
    sub_1890E40(a1, v19, v20, v14, v21);
    a4 = v9 - v14;
    a3 = v10;
    v5 = v22;
    a5 = v8 - v21;
    a2 = v18;
    a1 = v20;
    v6 = v23;
    v7 = v24;
  }
}

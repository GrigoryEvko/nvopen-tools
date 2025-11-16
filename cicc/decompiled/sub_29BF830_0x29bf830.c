// Function: sub_29BF830
// Address: 0x29bf830
//
void __fastcall sub_29BF830(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r14
  char *v12; // r15
  char *v13; // rax
  char *v14; // r10
  __int64 *v15; // r9
  char *v16; // r11
  char *v17; // rax
  __int64 *v18; // r9
  char *v19; // rax
  __int64 v20; // rdx
  char *v21; // [rsp-60h] [rbp-60h]
  __int64 *v22; // [rsp-58h] [rbp-58h]
  __int64 *v23; // [rsp-50h] [rbp-50h]
  __int64 *v24; // [rsp-50h] [rbp-50h]
  char *v25; // [rsp-50h] [rbp-50h]
  __int64 v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]
  __int64 v28; // [rsp-30h] [rbp-30h]
  __int64 v29; // [rsp-18h] [rbp-18h]
  __int64 v30; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v30 = v8;
    v29 = v7;
    v9 = a5;
    v28 = v6;
    if ( !a5 )
      break;
    v10 = a4;
    if ( a4 + a5 == 2 )
    {
      v20 = *(_QWORD *)a1;
      if ( *(_DWORD *)(*a6 + 16LL * *(_QWORD *)a2) < *(_DWORD *)(*a6 + 16LL * *(_QWORD *)a1) )
      {
        *(_QWORD *)a1 = *(_QWORD *)a2;
        *(_QWORD *)a2 = v20;
      }
      return;
    }
    v11 = a3;
    v23 = a6;
    if ( a4 > a5 )
    {
      v27 = a4 / 2;
      v19 = (char *)sub_29BF5B0(a2, a3, &a1[8 * (a4 / 2)], a6);
      v15 = v23;
      v12 = v19;
      v26 = (v19 - v14) >> 3;
    }
    else
    {
      v12 = &a2[8 * (a5 / 2)];
      v26 = a5 / 2;
      v13 = (char *)sub_29BF610(a1, (__int64)a2, v12, a6);
      v15 = v23;
      v16 = v13;
      v27 = (v13 - a1) >> 3;
    }
    v24 = v15;
    v21 = v16;
    v17 = sub_29BF670(v16, v14, v12);
    v18 = v24;
    v25 = v17;
    v22 = v18;
    sub_29BF830(a1, v21, v17, v27, v26);
    a3 = v11;
    a6 = v22;
    a4 = v10 - v27;
    a5 = v9 - v26;
    v6 = v28;
    a2 = v12;
    a1 = v25;
    v7 = v29;
    v8 = v30;
  }
}

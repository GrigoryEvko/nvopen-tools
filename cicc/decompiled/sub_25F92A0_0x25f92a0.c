// Function: sub_25F92A0
// Address: 0x25f92a0
//
void __fastcall sub_25F92A0(char *a1, char *a2, __int64 a3, signed __int64 a4, signed __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  char *v9; // r11
  __int64 v10; // rbx
  __int64 v11; // r15
  char *v12; // rax
  char *v13; // r11
  char *v14; // r9
  char *v15; // r10
  __int64 v16; // r14
  char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rax
  char *v22; // [rsp-58h] [rbp-58h]
  char *v23; // [rsp-50h] [rbp-50h]
  char *v24; // [rsp-48h] [rbp-48h]
  char *v25; // [rsp-48h] [rbp-48h]
  char *v26; // [rsp-48h] [rbp-48h]
  __int64 v27; // [rsp-40h] [rbp-40h]
  __int64 v28; // [rsp-30h] [rbp-30h]
  __int64 v29; // [rsp-18h] [rbp-18h]
  __int64 v30; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v30 = v7;
    v29 = v6;
    v8 = a5;
    v28 = v5;
    if ( !a5 )
      break;
    v9 = a2;
    v10 = a4;
    if ( a4 + a5 == 2 )
    {
      v18 = *(_QWORD *)a2;
      v19 = *(_QWORD *)a1;
      v20 = *((_QWORD *)a1 + 1);
      if ( *(unsigned int *)(v18 + 4) * 0x86BCA1AF286BCA1BLL * ((*((_QWORD *)v9 + 1) - v18) >> 3) > *(unsigned int *)(*(_QWORD *)a1 + 4LL) * 0x86BCA1AF286BCA1BLL * ((v20 - *(_QWORD *)a1) >> 3) )
      {
        *(_QWORD *)a1 = v18;
        v21 = *((_QWORD *)a1 + 2);
        *((_QWORD *)a1 + 1) = *((_QWORD *)v9 + 1);
        *((_QWORD *)a1 + 2) = *((_QWORD *)v9 + 2);
        *(_QWORD *)v9 = v19;
        *((_QWORD *)v9 + 1) = v20;
        *((_QWORD *)v9 + 2) = v21;
      }
      return;
    }
    v11 = a3;
    if ( a4 > a5 )
    {
      v16 = a4 / 2;
      v26 = &a1[8 * (a4 / 2) + 8 * ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v17 = (char *)sub_25F69C0(a2, a3, v26);
      v15 = v26;
      v14 = v17;
      v27 = 0xAAAAAAAAAAAAAAABLL * ((v17 - v13) >> 3);
    }
    else
    {
      v27 = a5 / 2;
      v24 = &a2[8 * (a5 / 2) + 8 * ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v12 = (char *)sub_25F6920(a1, (__int64)a2, v24);
      v14 = v24;
      v15 = v12;
      v16 = 0xAAAAAAAAAAAAAAABLL * ((v12 - a1) >> 3);
    }
    v22 = v14;
    v23 = v15;
    v25 = sub_25F77F0(v15, v13, v14);
    sub_25F92A0(a1, v23, v25, v16, v27);
    a4 = v10 - v16;
    a3 = v11;
    v5 = v28;
    a5 = v8 - v27;
    a2 = v22;
    a1 = v25;
    v6 = v29;
    v7 = v30;
  }
}

// Function: sub_2608140
// Address: 0x2608140
//
void __fastcall sub_2608140(char *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 *v11; // rax
  __int64 *v12; // r9
  __int64 *v13; // r10
  unsigned __int64 v14; // r8
  __int64 *v15; // r11
  __int64 v16; // r14
  __int64 *v17; // rax
  unsigned __int64 v18; // r8
  __int64 *v19; // [rsp-58h] [rbp-58h]
  __int64 *v20; // [rsp-50h] [rbp-50h]
  unsigned __int64 v21; // [rsp-48h] [rbp-48h]
  __int64 v22; // [rsp-40h] [rbp-40h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]
  __int64 *v24; // [rsp-40h] [rbp-40h]
  __int64 v25; // [rsp-30h] [rbp-30h]
  __int64 v26; // [rsp-18h] [rbp-18h]
  __int64 v27; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v27 = v7;
    v26 = v6;
    v8 = a5;
    v25 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      if ( *(_DWORD *)a2 < *(_DWORD *)a1 )
        sub_2607F90((__int64)a1, (__int64)a2);
      return;
    }
    v10 = a3;
    if ( a4 > a5 )
    {
      v16 = a4 / 2;
      v13 = (__int64 *)sub_25F67A0(a2, a3, (unsigned int *)&a1[152 * (a4 / 2)]);
      v14 = 0x86BCA1AF286BCA1BLL * (v13 - v12);
    }
    else
    {
      v22 = a5 / 2;
      v11 = (__int64 *)sub_25F6810(a1, (__int64)a2, (unsigned int *)&a2[19 * (a5 / 2)]);
      v14 = v22;
      v15 = v11;
      v16 = 0x86BCA1AF286BCA1BLL * (((char *)v11 - a1) >> 3);
    }
    v19 = v13;
    v23 = v14;
    v20 = v15;
    v17 = sub_25FED10(v15, v12, v13);
    v18 = v23;
    v24 = v17;
    v21 = v18;
    sub_2608140(a1, v20, v17, v16);
    a4 = v9 - v16;
    a3 = v10;
    v5 = v25;
    a2 = v19;
    a5 = v8 - v21;
    a1 = (char *)v24;
    v6 = v26;
    v7 = v27;
  }
}

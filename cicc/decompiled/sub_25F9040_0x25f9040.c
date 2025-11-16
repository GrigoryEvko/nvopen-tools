// Function: sub_25F9040
// Address: 0x25f9040
//
void __fastcall sub_25F9040(char *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
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
  __int64 *v16; // rax
  int v17; // eax
  unsigned __int64 v18; // rdx
  unsigned int v19; // r15d
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rbx
  unsigned int v23; // r15d
  unsigned int v24; // r15d
  int v25; // eax
  __int64 v26; // [rsp-58h] [rbp-58h]
  __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 *v28; // [rsp-50h] [rbp-50h]
  __int64 *v29; // [rsp-50h] [rbp-50h]
  __int64 *v30; // [rsp-48h] [rbp-48h]
  char *v31; // [rsp-48h] [rbp-48h]
  __int64 v32; // [rsp-48h] [rbp-48h]
  __int64 v33; // [rsp-40h] [rbp-40h]
  unsigned __int64 v34; // [rsp-40h] [rbp-40h]
  __int64 v35; // [rsp-30h] [rbp-30h]
  __int64 v36; // [rsp-20h] [rbp-20h]
  __int64 v37; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v37 = v7;
    v36 = v6;
    v8 = a5;
    v35 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v21 = *a2;
      v22 = *(_QWORD *)a1;
      v23 = *(_DWORD *)(*a2 + 32);
      if ( v23 > 0x40 )
      {
        v25 = sub_C444A0(v21 + 24);
        v18 = -1;
        if ( v23 - v25 <= 0x40 )
          v18 = **(_QWORD **)(v21 + 24);
      }
      else
      {
        v18 = *(_QWORD *)(v21 + 24);
      }
      v24 = *(_DWORD *)(v22 + 32);
      if ( v24 > 0x40 )
      {
        v34 = v18;
        v17 = sub_C444A0(v22 + 24);
        v18 = v34;
        v19 = v24 - v17;
        v20 = -1;
        if ( v19 <= 0x40 )
          v20 = **(_QWORD **)(v22 + 24);
      }
      else
      {
        v20 = *(_QWORD *)(v22 + 24);
      }
      if ( v20 > v18 )
      {
        *(_QWORD *)a1 = v21;
        *a2 = v22;
      }
      return;
    }
    if ( a4 > a5 )
    {
      v32 = a3;
      v14 = a4 / 2;
      v29 = (__int64 *)&a1[8 * (a4 / 2)];
      v16 = sub_25F6FF0(a2, a3, v29);
      v11 = v32;
      v13 = (char *)v29;
      v12 = (char *)v16;
      v33 = v16 - a2;
    }
    else
    {
      v27 = a3;
      v33 = a5 / 2;
      v30 = &a2[a5 / 2];
      v10 = (char *)sub_25F7100(a1, (__int64)a2, v30);
      v11 = v27;
      v12 = (char *)v30;
      v13 = v10;
      v14 = (v10 - a1) >> 3;
    }
    v26 = v11;
    v28 = (__int64 *)v12;
    v31 = v13;
    v15 = sub_25F8A40(v13, (char *)a2, v12);
    sub_25F9040(a1, v31, v15, v14, v33);
    a4 = v9 - v14;
    a1 = v15;
    v5 = v35;
    a5 = v8 - v33;
    a3 = v26;
    a2 = v28;
    v6 = v36;
    v7 = v37;
  }
}

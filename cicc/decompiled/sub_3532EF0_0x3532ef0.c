// Function: sub_3532EF0
// Address: 0x3532ef0
//
void __fastcall sub_3532EF0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r15
  __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 (__fastcall ***v17)(_QWORD); // r15
  int v18; // eax
  unsigned int v19; // r12d
  __int64 v20; // r15
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // [rsp-58h] [rbp-58h]
  __int64 v24; // [rsp-50h] [rbp-50h]
  _QWORD *v25; // [rsp-50h] [rbp-50h]
  _QWORD *v26; // [rsp-50h] [rbp-50h]
  _QWORD *v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-48h] [rbp-48h]
  __int64 v30; // [rsp-40h] [rbp-40h]
  __int64 v31; // [rsp-30h] [rbp-30h]
  __int64 v32; // [rsp-20h] [rbp-20h]
  __int64 v33; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v33 = v7;
    v32 = v6;
    v8 = a5;
    v31 = v5;
    if ( !a5 )
      break;
    v9 = a4;
    if ( a4 + a5 == 2 )
    {
      v17 = (__int64 (__fastcall ***)(_QWORD))*a2;
      v18 = (**(__int64 (__fastcall ***)(_QWORD))*a2)(*a2);
      LODWORD(v17) = *((_DWORD *)v17 + 10);
      v19 = (_DWORD)v17 * (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a1 + 8LL))(*a1) * v18;
      v20 = *a1;
      v21 = (**(__int64 (__fastcall ***)(_QWORD))*a1)(*a1);
      LODWORD(v20) = *(_DWORD *)(v20 + 40);
      if ( v19 > (unsigned int)v20 * (*(unsigned int (__fastcall **)(_QWORD))(*(_QWORD *)*a2 + 8LL))(*a2) * v21 )
      {
        v22 = *a1;
        *a1 = *a2;
        *a2 = v22;
      }
      return;
    }
    if ( a4 > a5 )
    {
      v29 = a3;
      v14 = a4 / 2;
      v26 = &a1[a4 / 2];
      v16 = sub_3532E20(a2, a3, v26);
      v11 = v29;
      v13 = (__int64)v26;
      v12 = (__int64)v16;
      v30 = v16 - a2;
    }
    else
    {
      v24 = a3;
      v30 = a5 / 2;
      v27 = &a2[a5 / 2];
      v10 = sub_3532D60(a1, (__int64)a2, v27);
      v11 = v24;
      v12 = (__int64)v27;
      v13 = (__int64)v10;
      v14 = v10 - a1;
    }
    v23 = v11;
    v25 = (_QWORD *)v12;
    v28 = v13;
    v15 = sub_35321D0(v13, (__int64)a2, v12);
    sub_3532EF0(a1, v28, v15, v14, v30);
    a4 = v9 - v14;
    a1 = (_QWORD *)v15;
    v5 = v31;
    a5 = v8 - v30;
    a3 = v23;
    a2 = v25;
    v6 = v32;
    v7 = v33;
  }
}

// Function: sub_2427DA0
// Address: 0x2427da0
//
void __fastcall sub_2427DA0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  _QWORD *v9; // r11
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // r9
  __int64 v15; // r10
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // esi
  bool v21; // cf
  _QWORD *v22; // [rsp-58h] [rbp-58h]
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-48h] [rbp-48h]
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-48h] [rbp-48h]
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
      v18 = *a2;
      v19 = *(_QWORD *)a1;
      v20 = *(_DWORD *)(*(_QWORD *)a1 + 32LL);
      v21 = *(_DWORD *)(v18 + 32) < v20;
      if ( *(_DWORD *)(v18 + 32) == v20 )
        v21 = *(_DWORD *)(v18 + 36) < *(_DWORD *)(v19 + 36);
      if ( v21 )
      {
        *(_QWORD *)a1 = v18;
        *v9 = v19;
      }
      return;
    }
    v11 = a3;
    if ( a4 > a5 )
    {
      v16 = a4 / 2;
      v26 = a1 + 8 * (a4 / 2);
      v17 = sub_2425A00((__int64)a2, a3, v26);
      v15 = v26;
      v14 = v17;
      v27 = (v17 - v13) >> 3;
    }
    else
    {
      v27 = a5 / 2;
      v24 = (__int64)&a2[a5 / 2];
      v12 = sub_2425A60(a1, (__int64)a2, v24);
      v14 = v24;
      v15 = v12;
      v16 = (v12 - a1) >> 3;
    }
    v22 = (_QWORD *)v14;
    v23 = v15;
    v25 = sub_24252C0(v15, v13, v14);
    sub_2427DA0(a1, v23, v25, v16, v27);
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

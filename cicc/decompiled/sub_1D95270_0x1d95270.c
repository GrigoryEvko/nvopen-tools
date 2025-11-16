// Function: sub_1D95270
// Address: 0x1d95270
//
void __fastcall sub_1D95270(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 *v11; // r14
  __int64 *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // r11
  __int64 v19; // r13
  __int64 *v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // r8d
  int v23; // esi
  __int64 v24; // rax
  unsigned int v25; // edi
  int v26; // ecx
  unsigned __int8 v27; // cl
  __int64 *v28; // [rsp-60h] [rbp-60h]
  __int64 v29; // [rsp-58h] [rbp-58h]
  __int64 v30; // [rsp-58h] [rbp-58h]
  __int64 *v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-50h] [rbp-50h]
  __int64 *v33; // [rsp-50h] [rbp-50h]
  __int64 v34; // [rsp-48h] [rbp-48h]
  __int64 v35; // [rsp-40h] [rbp-40h]
  __int64 v36; // [rsp-30h] [rbp-30h]
  __int64 v37; // [rsp-20h] [rbp-20h]
  __int64 v38; // [rsp-18h] [rbp-18h]
  __int64 v39; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v39 = v9;
    v38 = v8;
    v37 = v7;
    v10 = a5;
    v36 = v6;
    if ( !a5 )
      break;
    v11 = a1;
    v12 = a2;
    v13 = a4;
    if ( a4 + a5 == 2 )
    {
      v21 = *a2;
      v22 = *(_DWORD *)(*a2 + 8);
      v23 = *(_DWORD *)(*a2 + 12);
      if ( v22 == 7 )
        v23 = -(*(_DWORD *)(v21 + 16) + v23);
      v24 = *a1;
      v25 = *(_DWORD *)(*a1 + 8);
      v26 = *(_DWORD *)(*v11 + 12);
      if ( v25 == 7 )
        v26 = -(*(_DWORD *)(v24 + 16) + v26);
      if ( v23 > v26
        || v23 == v26
        && ((v27 = *(_BYTE *)(v21 + 20), (v27 & 1) == 0) && (*(_BYTE *)(v24 + 20) & 1) != 0
         || ((*(_BYTE *)(v24 + 20) ^ v27) & 1) == 0
         && (v22 < v25
          || v22 == v25
          && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v21 + 16LL) + 48LL) < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v24 + 16LL)
                                                                                + 48LL))) )
      {
        *v11 = v21;
        *v12 = v24;
      }
      return;
    }
    v29 = a6;
    v14 = a3;
    if ( a4 > a5 )
    {
      v35 = a4 / 2;
      v33 = &a1[a4 / 2];
      v20 = sub_1D92CF0(a2, a3, v33);
      v18 = (__int64)v33;
      v16 = v29;
      v17 = (__int64)v20;
      v34 = v20 - a2;
    }
    else
    {
      v34 = a5 / 2;
      v31 = &a2[a5 / 2];
      v15 = sub_1D92C20(a1, (__int64)a2, v31);
      v16 = v29;
      v17 = (__int64)v31;
      v18 = (__int64)v15;
      v35 = v15 - a1;
    }
    v28 = (__int64 *)v17;
    v32 = v16;
    v30 = v18;
    v19 = sub_1D919E0(v18, (__int64)a2, v17);
    sub_1D95270(a1, v30, v19, v35, v34, v32);
    a3 = v14;
    a6 = v32;
    a4 = v13 - v35;
    a5 = v10 - v34;
    v6 = v36;
    a1 = (__int64 *)v19;
    a2 = v28;
    v7 = v37;
    v8 = v38;
    v9 = v39;
  }
}

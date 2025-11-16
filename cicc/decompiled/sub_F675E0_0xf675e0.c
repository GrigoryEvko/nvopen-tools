// Function: sub_F675E0
// Address: 0xf675e0
//
unsigned __int64 __fastcall sub_F675E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // r8
  _QWORD *v12; // r9
  __int64 v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // r9
  __int64 v19; // r8
  _QWORD *v20; // r9
  __int64 v21; // r8
  _QWORD *v22; // r9
  __int64 v23; // r8
  _QWORD *v24; // r9
  __int64 v25; // r8
  _QWORD *v26; // r9
  __int64 v27; // r8
  _QWORD *v28; // r9
  unsigned __int64 result; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r8
  __int64 v33; // r8
  __int64 v34; // r8
  __int64 v35; // r8
  __int64 v36; // r8
  __int64 v37; // r8
  __int64 v38; // r8
  __int64 v39; // r8
  __int64 v40; // r8
  __int64 v41[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8662C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v41[0] = (__int64)&unk_4F8144C;
  v5 = &v3[v4];
  if ( v5 == sub_F66EB0(v3, (__int64)v5, v41) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v8 = *(_QWORD **)(a2 + 112);
  v9 = *(unsigned int *)(a2 + 120);
  v41[0] = (__int64)&unk_4F875EC;
  v10 = &v8[v9];
  if ( v10 == sub_F66EB0(v8, (__int64)v10, v41) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, (__int64)v12);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F875EC;
    v12 = *(_QWORD **)(a2 + 112);
    v40 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v40;
    v10 = &v12[v40];
  }
  v41[0] = (__int64)&unk_4F8670C;
  if ( v10 == sub_F66EB0(v12, (__int64)v10, v41) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, (__int64)v14);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F8670C;
    v14 = *(_QWORD **)(a2 + 112);
    v39 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v39;
    v10 = &v14[v39];
  }
  v41[0] = (__int64)&unk_4F86530;
  if ( v10 == sub_F66EB0(v14, (__int64)v10, v41) )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, (__int64)v16);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F86530;
    v16 = *(_QWORD **)(a2 + 112);
    v38 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v38;
    v10 = &v16[v38];
  }
  v41[0] = (__int64)&unk_4F86B74;
  if ( v10 == sub_F66EB0(v16, (__int64)v10, v41) )
  {
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v17 + 1, 8u, v17, (__int64)v18);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F86B74;
    v18 = *(_QWORD **)(a2 + 112);
    v37 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v37;
    v10 = &v18[v37];
  }
  v41[0] = (__int64)&unk_4F881C8;
  if ( v10 == sub_F66EB0(v18, (__int64)v10, v41) )
  {
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, (__int64)v20);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F881C8;
    v20 = *(_QWORD **)(a2 + 112);
    v36 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v36;
    v10 = &v20[v36];
  }
  v41[0] = (__int64)&unk_4F89B30;
  if ( v10 == sub_F66EB0(v20, (__int64)v10, v41) )
  {
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v21 + 1, 8u, v21, (__int64)v22);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F89B30;
    v22 = *(_QWORD **)(a2 + 112);
    v35 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v35;
    v10 = &v22[v35];
  }
  v41[0] = (__int64)&unk_4F90E2C;
  if ( v10 == sub_F66EB0(v22, (__int64)v10, v41) )
  {
    if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v23 + 1, 8u, v23, (__int64)v24);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F90E2C;
    v24 = *(_QWORD **)(a2 + 112);
    v34 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v34;
    v10 = &v24[v34];
  }
  v41[0] = (__int64)&unk_4F8BE8C;
  if ( v10 == sub_F66EB0(v24, (__int64)v10, v41) )
  {
    if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v25 + 1, 8u, v25, (__int64)v26);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F8BE8C;
    v26 = *(_QWORD **)(a2 + 112);
    v33 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v33;
    v10 = &v26[v33];
  }
  v41[0] = (__int64)&unk_4F8E808;
  if ( v10 == sub_F66EB0(v26, (__int64)v10, v41) )
  {
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v27 + 1, 8u, v27, (__int64)v28);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F8E808;
    v28 = *(_QWORD **)(a2 + 112);
    v32 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v32;
    v10 = &v28[v32];
  }
  v41[0] = (__int64)&unk_4F8F808;
  result = (unsigned __int64)sub_F66EB0(v28, (__int64)v10, v41);
  if ( v10 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v30 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v30 + 1, 8u, v30, v31);
      result = *(_QWORD *)(a2 + 112);
      v10 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F8F808;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

// Function: sub_11CD970
// Address: 0x11cd970
//
unsigned __int64 __fastcall sub_11CD970(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  _QWORD *v4; // rdi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  _QWORD *v8; // r9
  __int64 v9; // r8
  _QWORD *v10; // r9
  __int64 v11; // r8
  _QWORD *v12; // r9
  __int64 v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rdi
  __int64 v22; // r8
  _QWORD *v23; // rsi
  unsigned __int64 result; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r8
  __int64 v29; // r8
  __int64 v30; // r8
  __int64 v31; // r8
  __int64 v32; // r8
  __int64 v33; // r8
  __int64 v34[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v4 = *(_QWORD **)(a2 + 112);
  v34[0] = (__int64)&unk_4F8C12C;
  if ( v3 == sub_11CD750(v4, (__int64)v3, v34) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8C12C;
    v6 = *(_QWORD **)(a2 + 112);
    v27 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v27;
    v3 = &v6[v27];
  }
  v34[0] = (__int64)&unk_4F86530;
  if ( v3 == sub_11CD750(v6, (__int64)v3, v34) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, (__int64)v8);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F86530;
    v8 = *(_QWORD **)(a2 + 112);
    v33 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v33;
    v3 = &v8[v33];
  }
  v34[0] = (__int64)&unk_4F8670C;
  if ( v3 == sub_11CD750(v8, (__int64)v3, v34) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, (__int64)v10);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8670C;
    v10 = *(_QWORD **)(a2 + 112);
    v32 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v32;
    v3 = &v10[v32];
  }
  v34[0] = (__int64)&unk_4F86B74;
  if ( v3 == sub_11CD750(v10, (__int64)v3, v34) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, (__int64)v12);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F86B74;
    v12 = *(_QWORD **)(a2 + 112);
    v31 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v31;
    v3 = &v12[v31];
  }
  v34[0] = (__int64)&unk_4F881C8;
  if ( v3 == sub_11CD750(v12, (__int64)v3, v34) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, (__int64)v14);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F881C8;
    v14 = *(_QWORD **)(a2 + 112);
    v30 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v30;
    v3 = &v14[v30];
  }
  v34[0] = (__int64)&unk_4F89B30;
  if ( v3 == sub_11CD750(v14, (__int64)v3, v34) )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, (__int64)v16);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F89B30;
    v16 = *(_QWORD **)(a2 + 112);
    v29 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v29;
    v3 = &v16[v29];
  }
  v34[0] = (__int64)&unk_4F8E808;
  if ( v3 == sub_11CD750(v16, (__int64)v3, v34) )
  {
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v17 + 1, 8u, v17, (__int64)v18);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8E808;
    v18 = *(_QWORD **)(a2 + 112);
    v28 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v28;
    v3 = &v18[v28];
  }
  v34[0] = (__int64)&unk_4F8F808;
  if ( v3 == sub_11CD750(v18, (__int64)v3, v34) )
  {
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, v20);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v3 = &unk_4F8F808;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F876DC);
  v21 = *(_QWORD **)(a2 + 112);
  v22 = *(unsigned int *)(a2 + 120);
  v34[0] = (__int64)&unk_4F876DC;
  v23 = &v21[v22];
  result = (unsigned __int64)sub_11CD750(v21, (__int64)v23, v34);
  if ( v23 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v25 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v25 + 1, 8u, v25, v26);
      result = *(_QWORD *)(a2 + 112);
      v23 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v23 = &unk_4F876DC;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

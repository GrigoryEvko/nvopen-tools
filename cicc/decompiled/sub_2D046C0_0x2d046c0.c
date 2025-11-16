// Function: sub_2D046C0
// Address: 0x2d046c0
//
unsigned __int64 __fastcall sub_2D046C0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  __int64 v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rdi
  __int64 v14; // r8
  _QWORD *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rdi
  __int64 v19; // r8
  _QWORD *v20; // rsi
  unsigned __int64 result; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v24[0] = (__int64)&unk_4F875EC;
  v5 = &v3[v4];
  if ( v5 == sub_2D04150(v3, (__int64)v5, v24) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v8 = *(_QWORD **)(a2 + 112);
  v9 = *(unsigned int *)(a2 + 120);
  v24[0] = (__int64)&unk_4F8144C;
  v10 = &v8[v9];
  if ( v10 == sub_2D04150(v8, (__int64)v10, v24) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, v12);
      v10 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v10 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8D9B0);
  v13 = *(_QWORD **)(a2 + 112);
  v14 = *(unsigned int *)(a2 + 120);
  v24[0] = (__int64)&unk_4F8D9B0;
  v15 = &v13[v14];
  if ( v15 == sub_2D04150(v13, (__int64)v15, v24) )
  {
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v16 + 1, 8u, v16, v17);
      v15 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v15 = &unk_4F8D9B0;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F89C28);
  v18 = *(_QWORD **)(a2 + 112);
  v19 = *(unsigned int *)(a2 + 120);
  v24[0] = (__int64)&unk_4F89C28;
  v20 = &v18[v19];
  result = (unsigned __int64)sub_2D04150(v18, (__int64)v20, v24);
  if ( v20 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v22 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v22 + 1, 8u, v22, v23);
      result = *(_QWORD *)(a2 + 112);
      v20 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v20 = &unk_4F89C28;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

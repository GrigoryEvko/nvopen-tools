// Function: sub_287BBE0
// Address: 0x287bbe0
//
unsigned __int64 __fastcall sub_287BBE0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rdi
  __int64 v11; // r8
  _QWORD *v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rdi
  __int64 v16; // r8
  _QWORD *v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  unsigned __int64 result; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v26[0] = (__int64)&unk_4F875EC;
  v5 = &v3[v4];
  if ( v5 == sub_287BAB0(v3, (__int64)v5, v26) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    v7 = *(_QWORD **)(a2 + 112);
    v25 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v25;
    v5 = &v7[v25];
  }
  v26[0] = (__int64)&unk_4F8C12C;
  if ( v5 == sub_287BAB0(v7, (__int64)v5, v26) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, v9);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8C12C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8C12C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v10 = *(_QWORD **)(a2 + 112);
  v11 = *(unsigned int *)(a2 + 120);
  v26[0] = (__int64)&unk_4F8144C;
  v12 = &v10[v11];
  if ( v12 == sub_287BAB0(v10, (__int64)v12, v26) )
  {
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v13 + 1, 8u, v13, v14);
      v12 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v12 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F881C8);
  v15 = *(_QWORD **)(a2 + 112);
  v16 = *(unsigned int *)(a2 + 120);
  v26[0] = (__int64)&unk_4F881C8;
  v17 = &v15[v16];
  if ( v17 == sub_287BAB0(v15, (__int64)v17, v26) )
  {
    if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v18 + 1, 8u, v18, v19);
      v17 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v17 = &unk_4F881C8;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F89C28);
  v20 = *(_QWORD **)(a2 + 112);
  v21 = &v20[*(unsigned int *)(a2 + 120)];
  v26[0] = (__int64)&unk_4F8F808;
  result = (unsigned __int64)sub_287BAB0(v20, (__int64)v21, v26);
  if ( v21 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v23 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v23 + 1, 8u, v23, v24);
      result = *(_QWORD *)(a2 + 112);
      v21 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v21 = &unk_4F8F808;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

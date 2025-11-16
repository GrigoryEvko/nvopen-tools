// Function: sub_29A7340
// Address: 0x29a7340
//
unsigned __int64 __fastcall sub_29A7340(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rdi
  __int64 v8; // r8
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rdi
  __int64 v13; // r8
  _QWORD *v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rdi
  __int64 v18; // r8
  _QWORD *v19; // rsi
  unsigned __int64 result; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v23[0] = (__int64)&unk_4F8C12C;
  if ( v4 == sub_29A7280(v3, (__int64)v4, v23) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8C12C;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v7 = *(_QWORD **)(a2 + 112);
  v8 = *(unsigned int *)(a2 + 120);
  v23[0] = (__int64)&unk_4F875EC;
  v9 = &v7[v8];
  if ( v9 == sub_29A7280(v7, (__int64)v9, v23) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, v11);
      v9 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v9 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8C12C);
  sub_BB9660(a2, (__int64)&unk_4F881C8);
  v12 = *(_QWORD **)(a2 + 112);
  v13 = *(unsigned int *)(a2 + 120);
  v23[0] = (__int64)&unk_4F881C8;
  v14 = &v12[v13];
  if ( v14 == sub_29A7280(v12, (__int64)v14, v23) )
  {
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, v16);
      v14 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v14 = &unk_4F881C8;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v17 = *(_QWORD **)(a2 + 112);
  v18 = *(unsigned int *)(a2 + 120);
  v23[0] = (__int64)&unk_4F8144C;
  v19 = &v17[v18];
  result = (unsigned __int64)sub_29A7280(v17, (__int64)v19, v23);
  if ( v19 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v21 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v21 + 1, 8u, v21, v22);
      result = *(_QWORD *)(a2 + 112);
      v19 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v19 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

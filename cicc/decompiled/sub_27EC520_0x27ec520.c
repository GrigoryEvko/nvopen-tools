// Function: sub_27EC520
// Address: 0x27ec520
//
unsigned __int64 __fastcall sub_27EC520(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  _QWORD *v4; // rsi
  __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rdi
  __int64 v10; // r8
  _QWORD *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rsi
  _QWORD *v15; // rdi
  __int64 v16; // r8
  _QWORD *v17; // r9
  unsigned __int64 result; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r8
  __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 112);
  v4 = &v3[*(unsigned int *)(a2 + 120)];
  v23[0] = (__int64)&unk_4F8144C;
  if ( v4 == sub_27EBF10(v3, (__int64)v4, v23) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v5, (__int64)v6);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F8144C;
    v6 = *(_QWORD **)(a2 + 112);
    v21 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v21;
    v4 = &v6[v21];
  }
  v23[0] = (__int64)&unk_4F875EC;
  if ( v4 == sub_27EBF10(v6, (__int64)v4, v23) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v7 + 1, 8u, v7, v8);
      v4 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v4 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F8F808);
  v9 = *(_QWORD **)(a2 + 112);
  v10 = *(unsigned int *)(a2 + 120);
  v23[0] = (__int64)&unk_4F8F808;
  v11 = &v9[v10];
  if ( v11 == sub_27EBF10(v9, (__int64)v11, v23) )
  {
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, v13);
      v11 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v11 = &unk_4F8F808;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F89C28);
  sub_BB9660(a2, (__int64)&unk_4F8662C);
  sub_F6D720(a2);
  sub_1027A20(a2);
  v14 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  v15 = *(_QWORD **)(a2 + 112);
  v23[0] = (__int64)&unk_4F8EE48;
  if ( v14 == sub_27EBF10(v15, (__int64)v14, v23) )
  {
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v16 + 1, 8u, v16, (__int64)v17);
      v14 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v14 = &unk_4F8EE48;
    v17 = *(_QWORD **)(a2 + 112);
    v22 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v22;
    v14 = &v17[v22];
  }
  v23[0] = (__int64)&unk_4F8EE50;
  result = (unsigned __int64)sub_27EBF10(v17, (__int64)v14, v23);
  if ( v14 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v19 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v19 + 1, 8u, v19, v20);
      result = *(_QWORD *)(a2 + 112);
      v14 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v14 = &unk_4F8EE50;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

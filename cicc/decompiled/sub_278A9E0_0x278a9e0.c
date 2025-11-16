// Function: sub_278A9E0
// Address: 0x278a9e0
//
__int64 __fastcall sub_278A9E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  _QWORD *v4; // r9
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  _QWORD *v9; // r9
  __int64 v10; // r8
  _QWORD *v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rdi
  _QWORD *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 result; // rax
  __int64 v19; // r8
  __int64 v20; // r8
  __int64 v21; // r8
  __int64 v22[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a1 + 176;
  sub_BB9660(a2, (__int64)&unk_4F8662C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  if ( (unsigned __int8)sub_278A9A0(a1 + 176) )
    sub_BB9660(a2, (__int64)&unk_4F8EE5C);
  sub_BB9660(a2, (__int64)&unk_4F86530);
  v4 = *(_QWORD **)(a2 + 112);
  v22[0] = (__int64)&unk_4F8144C;
  v5 = &v4[*(unsigned int *)(a2 + 120)];
  if ( v5 == sub_27896B0(v4, (__int64)v5, v22) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8144C;
    v7 = *(_QWORD **)(a2 + 112);
    v21 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v21;
    v5 = &v7[v21];
  }
  v22[0] = (__int64)&unk_4F86B74;
  if ( v5 == sub_27896B0(v7, (__int64)v5, v22) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, (__int64)v9);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F86B74;
    v9 = *(_QWORD **)(a2 + 112);
    v20 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v20;
    v5 = &v9[v20];
  }
  v22[0] = (__int64)&unk_4F6D3F0;
  if ( v5 == sub_27896B0(v9, (__int64)v5, v22) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, (__int64)v11);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F6D3F0;
    v11 = *(_QWORD **)(a2 + 112);
    v19 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v19;
    v5 = &v11[v19];
  }
  v22[0] = (__int64)&unk_4F875EC;
  if ( v5 == sub_27896B0(v11, (__int64)v5, v22) )
  {
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, v13);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8FAE4);
  v14 = *(_QWORD **)(a2 + 112);
  v15 = &v14[*(unsigned int *)(a2 + 120)];
  v22[0] = (__int64)&unk_4F8F808;
  if ( v15 == sub_27896B0(v14, (__int64)v15, v22) )
  {
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v16 + 1, 8u, v16, v17);
      v15 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v15 = &unk_4F8F808;
    ++*(_DWORD *)(a2 + 120);
  }
  result = sub_278A9C0(v3);
  if ( (_BYTE)result )
    return sub_BB9660(a2, (__int64)&unk_4F8F808);
  return result;
}

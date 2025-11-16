// Function: sub_2F81300
// Address: 0x2f81300
//
unsigned __int64 __fastcall sub_2F81300(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  _QWORD *v9; // r9
  __int64 v10; // r8
  _QWORD *v11; // r9
  __int64 v12; // r8
  _QWORD *v13; // r9
  unsigned __int64 result; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20; // r8
  __int64 v21[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_4F6D3F0);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v21[0] = (__int64)&unk_4F6D3F0;
  v5 = &v3[v4];
  if ( v5 == sub_2F81240(v3, (__int64)v5, v21) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F6D3F0;
    v7 = *(_QWORD **)(a2 + 112);
    v17 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v17;
    v5 = &v7[v17];
  }
  v21[0] = (__int64)&unk_4F881C8;
  if ( v5 == sub_2F81240(v7, (__int64)v5, v21) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, (__int64)v9);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F881C8;
    v9 = *(_QWORD **)(a2 + 112);
    v20 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v20;
    v5 = &v9[v20];
  }
  v21[0] = (__int64)&unk_4F86530;
  if ( v5 == sub_2F81240(v9, (__int64)v5, v21) )
  {
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v10 + 1, 8u, v10, (__int64)v11);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F86530;
    v11 = *(_QWORD **)(a2 + 112);
    v19 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v19;
    v5 = &v11[v19];
  }
  v21[0] = (__int64)&unk_4F8FAE4;
  if ( v5 == sub_2F81240(v11, (__int64)v5, v21) )
  {
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v12 + 1, 8u, v12, (__int64)v13);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8FAE4;
    v13 = *(_QWORD **)(a2 + 112);
    v18 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v18;
    v5 = &v13[v18];
  }
  v21[0] = (__int64)&unk_4F86B74;
  result = (unsigned __int64)sub_2F81240(v13, (__int64)v5, v21);
  if ( v5 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v15 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v15 + 1, 8u, v15, v16);
      result = *(_QWORD *)(a2 + 112);
      v5 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F86B74;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

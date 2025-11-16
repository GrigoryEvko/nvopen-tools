// Function: sub_CF46A0
// Address: 0xcf46a0
//
unsigned __int64 __fastcall sub_CF46A0(__int64 a1, __int64 a2)
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
  unsigned __int64 result; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19; // r8
  __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  *(_BYTE *)(a2 + 160) = 1;
  sub_BB96F0(a2, (__int64)&unk_4F8670C);
  sub_BB96F0(a2, (__int64)&unk_4F6D3F0);
  v3 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
  v4 = *(_QWORD **)(a2 + 144);
  v20[0] = (__int64)&unk_4F89B44;
  if ( v3 == sub_CF44E0(v4, (__int64)v3, v20) )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v5 + 1, 8u, v5, (__int64)v6);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v3 = &unk_4F89B44;
    v6 = *(_QWORD **)(a2 + 144);
    v16 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v16;
    v3 = &v6[v16];
  }
  v20[0] = (__int64)&unk_4F89FAC;
  if ( v3 == sub_CF44E0(v6, (__int64)v3, v20) )
  {
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v7 + 1, 8u, v7, (__int64)v8);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v3 = &unk_4F89FAC;
    v8 = *(_QWORD **)(a2 + 144);
    v19 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v19;
    v3 = &v8[v19];
  }
  v20[0] = (__int64)&unk_4F86B74;
  if ( v3 == sub_CF44E0(v8, (__int64)v3, v20) )
  {
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v9 + 1, 8u, v9, (__int64)v10);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v3 = &unk_4F86B74;
    v10 = *(_QWORD **)(a2 + 144);
    v18 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v18;
    v3 = &v10[v18];
  }
  v20[0] = (__int64)&unk_4F89B30;
  if ( v3 == sub_CF44E0(v10, (__int64)v3, v20) )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v11 + 1, 8u, v11, (__int64)v12);
      v3 = (_QWORD *)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v3 = &unk_4F89B30;
    v12 = *(_QWORD **)(a2 + 144);
    v17 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
    *(_DWORD *)(a2 + 152) = v17;
    v3 = &v12[v17];
  }
  v20[0] = (__int64)&unk_4F86538;
  result = (unsigned __int64)sub_CF44E0(v12, (__int64)v3, v20);
  if ( v3 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 156);
    if ( v14 + 1 > result )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v14 + 1, 8u, v14, v15);
      result = *(_QWORD *)(a2 + 144);
      v3 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v3 = &unk_4F86538;
    ++*(_DWORD *)(a2 + 152);
  }
  return result;
}

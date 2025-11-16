// Function: sub_29D35F0
// Address: 0x29d35f0
//
unsigned __int64 __fastcall sub_29D35F0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  __int64 v8; // r8
  _QWORD *v9; // r9
  unsigned __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F92384);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v15[0] = (__int64)&unk_4F8144C;
  v5 = &v3[v4];
  if ( v5 == sub_29D3530(v3, (__int64)v5, v15) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8144C;
    v7 = *(_QWORD **)(a2 + 112);
    v13 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v13;
    v5 = &v7[v13];
  }
  v15[0] = (__int64)&unk_4F92384;
  if ( v5 == sub_29D3530(v7, (__int64)v5, v15) )
  {
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v8 + 1, 8u, v8, (__int64)v9);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F92384;
    v9 = *(_QWORD **)(a2 + 112);
    v14 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v14;
    v5 = &v9[v14];
  }
  v15[0] = (__int64)&unk_4F875EC;
  result = (unsigned __int64)sub_29D3530(v9, (__int64)v5, v15);
  if ( v5 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v11 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v11 + 1, 8u, v11, v12);
      result = *(_QWORD *)(a2 + 112);
      v5 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

// Function: sub_2A86870
// Address: 0x2a86870
//
unsigned __int64 __fastcall sub_2A86870(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r9
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  _QWORD *v7; // r9
  unsigned __int64 result; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F875EC);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v12[0] = (__int64)&unk_4F875EC;
  v5 = &v3[v4];
  if ( v5 == sub_2A867B0(v3, (__int64)v5, v12) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, (__int64)v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    v7 = *(_QWORD **)(a2 + 112);
    v11 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
    *(_DWORD *)(a2 + 120) = v11;
    v5 = &v7[v11];
  }
  v12[0] = (__int64)&unk_4F8144C;
  result = (unsigned __int64)sub_2A867B0(v7, (__int64)v5, v12);
  if ( v5 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 124);
    if ( v9 + 1 > result )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v9 + 1, 8u, v9, v10);
      result = *(_QWORD *)(a2 + 112);
      v5 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F8144C;
    ++*(_DWORD *)(a2 + 120);
  }
  return result;
}

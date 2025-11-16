// Function: sub_26178E0
// Address: 0x26178e0
//
unsigned __int64 __fastcall sub_26178E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_BB9660(a2, (__int64)&unk_4F8BE8C);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  sub_BB9660(a2, (__int64)&unk_4F875EC);
  v3 = *(_QWORD **)(a2 + 112);
  v4 = *(unsigned int *)(a2 + 120);
  v13[0] = (__int64)&unk_4F875EC;
  v5 = &v3[v4];
  if ( v5 == sub_26176F0(v3, (__int64)v5, v13) )
  {
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
    {
      sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v6, v7);
      v5 = (_QWORD *)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
    }
    *v5 = &unk_4F875EC;
    ++*(_DWORD *)(a2 + 120);
  }
  sub_BB9660(a2, (__int64)&unk_4F8C12C);
  v8 = *(_QWORD **)(a2 + 144);
  v9 = &v8[*(unsigned int *)(a2 + 152)];
  v13[0] = (__int64)&unk_4F8662C;
  result = (unsigned __int64)sub_26176F0(v8, (__int64)v9, v13);
  if ( v9 == (_QWORD *)result )
  {
    result = *(unsigned int *)(a2 + 156);
    if ( v11 + 1 > result )
    {
      sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v11 + 1, 8u, v11, v12);
      result = *(_QWORD *)(a2 + 144);
      v9 = (_QWORD *)(result + 8LL * *(unsigned int *)(a2 + 152));
    }
    *v9 = &unk_4F8662C;
    ++*(_DWORD *)(a2 + 152);
  }
  return result;
}

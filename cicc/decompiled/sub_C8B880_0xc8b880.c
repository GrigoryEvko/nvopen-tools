// Function: sub_C8B880
// Address: 0xc8b880
//
int __fastcall sub_C8B880(__int64 a1, int a2, char a3)
{
  int v3; // r9d
  __int64 v4; // rax
  const void *v5; // r13
  size_t v6; // rcx
  size_t v8; // [rsp+0h] [rbp-20h]

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 136);
  v5 = *(const void **)v4;
  v6 = *(_QWORD *)(v4 + 8);
  if ( a3
    || !*(_BYTE *)(a1 + 184)
    || *(_QWORD *)(a1 + 160) != v6
    || v6
    && (v8 = *(_QWORD *)(v4 + 8), LODWORD(v4) = memcmp(*(const void **)(a1 + 152), v5, v6), v6 = v8, v3 = a2, (_DWORD)v4) )
  {
    LODWORD(v4) = sub_C57130(a1 + 192, a1, v5, v6, a1 + 144, v3);
  }
  return v4;
}

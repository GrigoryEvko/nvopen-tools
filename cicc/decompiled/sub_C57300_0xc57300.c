// Function: sub_C57300
// Address: 0xc57300
//
void __fastcall sub_C57300(__int64 a1, int a2, char a3)
{
  int v3; // r9d
  const void *v4; // r13
  size_t v5; // rcx
  int v6; // eax
  size_t v7; // [rsp+0h] [rbp-20h]

  v3 = a2;
  v4 = *(const void **)(a1 + 136);
  v5 = *(_QWORD *)(a1 + 144);
  if ( a3
    || !*(_BYTE *)(a1 + 208)
    || v5 != *(_QWORD *)(a1 + 184)
    || v5 && (v7 = *(_QWORD *)(a1 + 144), v6 = memcmp(*(const void **)(a1 + 176), v4, v5), v5 = v7, v3 = a2, v6) )
  {
    sub_C57130(a1 + 216, a1, v4, v5, a1 + 168, v3);
  }
}

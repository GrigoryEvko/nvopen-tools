// Function: sub_16B4A00
// Address: 0x16b4a00
//
void __fastcall sub_16B4A00(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r9
  const char *v4; // r13
  size_t v5; // rcx
  int v6; // eax
  size_t v7; // [rsp+0h] [rbp-20h]

  v3 = a2;
  if ( a3 )
  {
    v4 = *(const char **)(a1 + 160);
    v5 = *(_QWORD *)(a1 + 168);
LABEL_4:
    sub_16B4870(a1 + 240, (char *)a1, v4, v5, a1 + 192, v3);
    return;
  }
  if ( *(_BYTE *)(a1 + 232) )
  {
    v4 = *(const char **)(a1 + 160);
    v5 = *(_QWORD *)(a1 + 168);
    if ( *(_QWORD *)(a1 + 208) != v5 )
      goto LABEL_4;
    if ( v5 )
    {
      v7 = *(_QWORD *)(a1 + 168);
      v6 = memcmp(*(const void **)(a1 + 200), v4, v5);
      v5 = v7;
      v3 = a2;
      if ( v6 )
        goto LABEL_4;
    }
  }
}

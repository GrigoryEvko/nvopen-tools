// Function: sub_16D5F60
// Address: 0x16d5f60
//
int __fastcall sub_16D5F60(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r9
  __int64 v4; // rax
  const char *v5; // r13
  size_t v6; // rcx
  size_t v8; // [rsp+0h] [rbp-20h]

  v3 = a2;
  v4 = *(_QWORD *)(a1 + 160);
  if ( a3 )
  {
    v5 = *(const char **)v4;
    v6 = *(_QWORD *)(v4 + 8);
LABEL_4:
    LODWORD(v4) = sub_16B4870(a1 + 216, (char *)a1, v5, v6, a1 + 168, v3);
    return v4;
  }
  if ( *(_BYTE *)(a1 + 208) )
  {
    v5 = *(const char **)v4;
    v6 = *(_QWORD *)(v4 + 8);
    if ( *(_QWORD *)(a1 + 184) != v6 )
      goto LABEL_4;
    if ( v6 )
    {
      v8 = *(_QWORD *)(v4 + 8);
      LODWORD(v4) = memcmp(*(const void **)(a1 + 176), v5, v6);
      v6 = v8;
      v3 = a2;
      if ( (_DWORD)v4 )
        goto LABEL_4;
    }
  }
  return v4;
}

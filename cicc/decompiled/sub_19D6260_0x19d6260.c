// Function: sub_19D6260
// Address: 0x19d6260
//
char __fastcall sub_19D6260(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rdi
  const char *v5; // rax
  __int64 v6; // rdi
  const char *v7; // r15
  size_t v8; // rdx
  size_t v9; // rbx
  size_t v10; // rdx
  const char *v11; // rsi
  size_t v12; // r14
  int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi

  v4 = *a1;
  if ( v4 )
    v4 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
  v5 = sub_1649960(v4);
  v6 = *a2;
  v7 = v5;
  v9 = v8;
  if ( *a2 )
    v6 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
  v11 = sub_1649960(v6);
  v12 = v10;
  if ( v9 <= v10 )
  {
    if ( !v9 || (v13 = memcmp(v7, v11, v9)) == 0 )
    {
      if ( v9 != v12 )
        goto LABEL_9;
      v15 = *a1;
      v16 = *a2;
      if ( *a1 )
      {
        v17 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        if ( v16 )
        {
          if ( v17 != *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)) )
          {
LABEL_18:
            v15 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
            if ( !v16 )
              goto LABEL_14;
            v14 = *(_DWORD *)(v16 + 20);
LABEL_20:
            LOBYTE(v14) = *(_QWORD *)(v16 - 24LL * (v14 & 0xFFFFFFF)) > v15;
            return v14;
          }
        }
        else if ( v17 )
        {
          goto LABEL_18;
        }
      }
      else if ( v16 )
      {
        v14 = *(_DWORD *)(v16 + 20);
        if ( *(_QWORD *)(v16 - 24LL * (v14 & 0xFFFFFFF)) )
          goto LABEL_20;
      }
      return (unsigned int)sub_16AEA10((__int64)(a1 + 2), (__int64)(a2 + 2)) >> 31;
    }
LABEL_13:
    if ( v13 < 0 )
      goto LABEL_10;
LABEL_14:
    LOBYTE(v14) = 0;
    return v14;
  }
  if ( !v10 )
    goto LABEL_14;
  v13 = memcmp(v7, v11, v10);
  if ( v13 )
    goto LABEL_13;
LABEL_9:
  if ( v9 >= v12 )
    goto LABEL_14;
LABEL_10:
  LOBYTE(v14) = 1;
  return v14;
}

// Function: sub_167FE60
// Address: 0x167fe60
//
__int64 __fastcall sub_167FE60(__int64 a1, char *a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rcx
  unsigned int v5; // r8d
  size_t v7; // rbx
  int v8; // r10d
  int v9; // r9d
  unsigned int i; // r13d
  __int64 v11; // r14
  int v12; // eax
  unsigned int v13; // r13d
  const void *v15; // rsi
  bool v16; // dl
  int v17; // eax
  int v18; // [rsp+Ch] [rbp-44h]
  unsigned int v19; // [rsp+10h] [rbp-40h]
  int v20; // [rsp+14h] [rbp-3Ch]
  __int64 v21; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 )
  {
    v5 = v3 - 1;
    v7 = (unsigned int)a3;
    v8 = 1;
    v9 = HIDWORD(a3);
    for ( i = (v3 - 1) & HIDWORD(a3); ; i = v5 & v13 )
    {
      v11 = v4 + 24LL * i;
      v12 = *(_DWORD *)(v11 + 12);
      if ( v9 != v12 )
        goto LABEL_4;
      v15 = *(const void **)v11;
      v16 = a2 + 1 == 0;
      if ( *(_QWORD *)v11 == -1 )
        break;
      v16 = a2 + 2 == 0;
      if ( v15 == (const void *)-2LL )
        break;
      if ( *(_DWORD *)(v11 + 8) == v7 )
      {
        v18 = v8;
        v19 = v5;
        v20 = v9;
        v21 = v4;
        if ( !v7 )
          return *(_QWORD *)(v11 + 16);
        v17 = memcmp(a2, v15, v7);
        v4 = v21;
        v9 = v20;
        v5 = v19;
        v8 = v18;
        if ( !v17 )
          return *(_QWORD *)(v11 + 16);
      }
LABEL_5:
      v13 = v8 + i;
      ++v8;
    }
    if ( v16 )
      return *(_QWORD *)(v11 + 16);
LABEL_4:
    if ( !v12 && *(_QWORD *)v11 == -1 )
      goto LABEL_7;
    goto LABEL_5;
  }
LABEL_7:
  v11 = v4 + 24LL * v3;
  return *(_QWORD *)(v11 + 16);
}

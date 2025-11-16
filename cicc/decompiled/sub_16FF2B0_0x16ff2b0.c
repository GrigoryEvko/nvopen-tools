// Function: sub_16FF2B0
// Address: 0x16ff2b0
//
_DWORD *__fastcall sub_16FF2B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rdi
  unsigned __int64 v5; // rsi
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rbx
  size_t v10; // rdx
  int v11; // eax
  bool v12; // sf
  unsigned __int64 v13; // r12
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // edi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _DWORD *result; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  int v28; // eax
  _QWORD *v29; // [rsp+8h] [rbp-98h]
  _QWORD *v30; // [rsp+18h] [rbp-88h]
  const __m128i *v31; // [rsp+28h] [rbp-78h] BYREF
  const char *v32; // [rsp+30h] [rbp-70h] BYREF
  __int64 v33; // [rsp+38h] [rbp-68h]
  _QWORD *v34; // [rsp+48h] [rbp-58h]
  _QWORD v35[9]; // [rsp+58h] [rbp-48h] BYREF

  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  v3 = (_QWORD *)(a1 + 128);
  v4 = a1 + 120;
  *(_QWORD *)(v4 - 120) = a2;
  v5 = (unsigned __int64)v3;
  v6 = v3;
  *(_QWORD *)(v4 - 112) = 0;
  *(_QWORD *)(v4 - 104) = 0;
  *(_QWORD *)(v4 - 40) = 0;
  *(_QWORD *)(v4 - 32) = 0;
  *(_QWORD *)(v4 - 24) = 1;
  *(_QWORD *)(v4 - 8) = 0;
  *(_DWORD *)(v4 + 8) = 0;
  *(_QWORD *)(v4 + 16) = 0;
  *(_QWORD *)(v4 + 24) = v3;
  *(_QWORD *)(v4 + 32) = v3;
  *(_QWORD *)(v4 + 40) = 0;
  v31 = (const __m128i *)&v32;
  v32 = "!";
  v30 = v3;
  v33 = 1;
  v29 = (_QWORD *)v4;
  v7 = sub_16FF170((_QWORD *)v4, v3, &v31);
  *(_QWORD *)(v7 + 48) = "!";
  *(_QWORD *)(v7 + 56) = 1;
  v9 = *(_QWORD *)(a1 + 136);
  v32 = (const char *)&unk_3F6A4C4;
  v33 = 2;
  if ( !v9 )
  {
    v6 = v30;
    goto LABEL_16;
  }
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v9 + 40);
      if ( v13 <= 2 )
        break;
      v5 = (unsigned __int64)&unk_3F6A4C4;
      v14 = memcmp(*(const void **)(v9 + 32), &unk_3F6A4C4, 2u);
      v12 = v14 < 0;
      if ( v14 )
        goto LABEL_9;
LABEL_5:
      if ( v13 <= 1 )
        goto LABEL_10;
LABEL_6:
      v6 = (_QWORD *)v9;
      v9 = *(_QWORD *)(v9 + 16);
      if ( !v9 )
        goto LABEL_11;
    }
    v10 = *(_QWORD *)(v9 + 40);
    if ( !v13 )
      goto LABEL_10;
    v5 = (unsigned __int64)&unk_3F6A4C4;
    v11 = memcmp(*(const void **)(v9 + 32), &unk_3F6A4C4, v10);
    v12 = v11 < 0;
    if ( !v11 )
      goto LABEL_5;
LABEL_9:
    if ( !v12 )
      goto LABEL_6;
LABEL_10:
    v9 = *(_QWORD *)(v9 + 24);
  }
  while ( v9 );
LABEL_11:
  if ( v30 == v6 )
    goto LABEL_16;
  v15 = v6[5];
  v16 = v6[4];
  if ( v15 <= 1 )
  {
    if ( !v15 )
      goto LABEL_17;
    v28 = *(unsigned __int8 *)v16;
    v5 = (unsigned int)(33 - v28);
    if ( v28 == 33 )
      goto LABEL_17;
  }
  else
  {
    v10 = 33;
    v17 = *(unsigned __int8 *)v16;
    v5 = (unsigned int)(33 - v17);
    if ( v17 == 33 )
    {
      v16 = *(unsigned __int8 *)(v16 + 1);
      v5 = (unsigned int)(33 - v16);
      if ( (_DWORD)v16 == 33 )
      {
        if ( v15 != 2 )
          goto LABEL_16;
        goto LABEL_17;
      }
    }
  }
  if ( (v5 & 0x80000000) != 0LL )
  {
LABEL_16:
    v5 = (unsigned __int64)v6;
    v31 = (const __m128i *)&v32;
    v6 = (_QWORD *)sub_16FF170(v29, v6, &v31);
  }
LABEL_17:
  v6[7] = 18;
  v6[6] = "tag:yaml.org,2002:";
  if ( (unsigned __int8)sub_16FF0B0((__int64 **)a1, (_BYTE *)v5, v10, v16, v8) )
  {
    sub_16FC2A0((unsigned __int64 **)a1, 5, v18, v19, v20);
    result = (_DWORD *)sub_16FC330((__int64 **)a1, 5u, v25, v26, v27);
    if ( *result == 5 )
      goto LABEL_21;
  }
  else
  {
    result = (_DWORD *)sub_16FC330((__int64 **)a1, v5, v18, v19, v20);
    if ( *result != 5 )
      return result;
LABEL_21:
    sub_16FC210((__int64)&v32, (unsigned __int64 **)a1, v22, v23, v24);
    result = v35;
    if ( v34 != v35 )
      return (_DWORD *)j_j___libc_free_0(v34, v35[0] + 1LL);
  }
  return result;
}

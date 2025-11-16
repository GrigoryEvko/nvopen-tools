// Function: sub_CAFBE0
// Address: 0xcafbe0
//
_DWORD *__fastcall sub_CAFBE0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rdi
  _BYTE *v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r15
  size_t v12; // rdx
  int v13; // eax
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rbx
  size_t v16; // rdx
  int v17; // eax
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
  _QWORD *v28; // [rsp+18h] [rbp-88h]
  const __m128i *v29; // [rsp+28h] [rbp-78h] BYREF
  void *v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h]
  _QWORD *v32; // [rsp+48h] [rbp-58h]
  _QWORD v33[9]; // [rsp+58h] [rbp-48h] BYREF

  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  v3 = (_QWORD *)(a1 + 120);
  v4 = a1 + 112;
  *(_QWORD *)(v4 - 112) = a2;
  v5 = v3;
  v6 = v3;
  *(_QWORD *)(v4 - 104) = 0;
  *(_QWORD *)(v4 - 96) = 0;
  *(_QWORD *)(v4 - 32) = 0;
  *(_QWORD *)(v4 - 24) = 0;
  *(_QWORD *)(v4 - 16) = 1;
  *(_QWORD *)(v4 - 8) = 0;
  *(_DWORD *)(v4 + 8) = 0;
  *(_QWORD *)(v4 + 16) = 0;
  *(_QWORD *)(v4 + 24) = v3;
  *(_QWORD *)(v4 + 32) = v3;
  *(_QWORD *)(v4 + 40) = 0;
  v29 = (const __m128i *)&v30;
  v30 = &unk_3F6A4C5;
  v28 = v3;
  v31 = 1;
  v7 = sub_CAFAC0((_QWORD *)v4, v3, &v29);
  *(_QWORD *)(v7 + 48) = &unk_3F6A4C5;
  *(_QWORD *)(v7 + 56) = 1;
  v11 = *(_QWORD *)(a1 + 128);
  v30 = &unk_3F6A4C4;
  v31 = 2;
  if ( !v11 )
  {
    v6 = v28;
    goto LABEL_17;
  }
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v11 + 40);
      if ( !v14 )
      {
LABEL_9:
        v11 = *(_QWORD *)(v11 + 24);
        goto LABEL_10;
      }
      v12 = 2;
      v5 = &unk_3F6A4C4;
      if ( v14 <= 2 )
        v12 = *(_QWORD *)(v11 + 40);
      v13 = memcmp(*(const void **)(v11 + 32), &unk_3F6A4C4, v12);
      if ( v13 )
        break;
      if ( v14 == 1 )
        goto LABEL_9;
LABEL_7:
      v6 = (_QWORD *)v11;
      v11 = *(_QWORD *)(v11 + 16);
      if ( !v11 )
        goto LABEL_11;
    }
    if ( v13 >= 0 )
      goto LABEL_7;
    v11 = *(_QWORD *)(v11 + 24);
LABEL_10:
    ;
  }
  while ( v11 );
LABEL_11:
  if ( v28 == v6 )
    goto LABEL_17;
  v15 = v6[5];
  if ( v15 )
  {
    v16 = 2;
    v5 = (_BYTE *)v6[4];
    if ( v15 <= 2 )
      v16 = v6[5];
    v17 = memcmp(&unk_3F6A4C4, v5, v16);
    if ( v17 )
    {
      if ( v17 < 0 )
      {
LABEL_17:
        v5 = v6;
        v29 = (const __m128i *)&v30;
        v6 = (_QWORD *)sub_CAFAC0((_QWORD *)v4, v6, &v29);
      }
    }
    else if ( v15 > 2 )
    {
      goto LABEL_17;
    }
  }
  v6[7] = 18;
  v6[6] = "tag:yaml.org,2002:";
  if ( (unsigned __int8)sub_CAFA00((__int64 **)a1, v5, v8, v9, v10) )
  {
    sub_CAD710((unsigned __int64 **)a1, 5, v18, v19, v20);
    result = (_DWORD *)sub_CAD7A0((__int64 **)a1, 5u, v25, v26, v27);
    if ( *result == 5 )
      goto LABEL_24;
  }
  else
  {
    result = (_DWORD *)sub_CAD7A0((__int64 **)a1, (unsigned __int64)v5, v18, v19, v20);
    if ( *result != 5 )
      return result;
LABEL_24:
    sub_CAD680((__int64)&v30, (unsigned __int64 **)a1, v22, v23, v24);
    result = v33;
    if ( v32 != v33 )
      return (_DWORD *)j_j___libc_free_0(v32, v33[0] + 1LL);
  }
  return result;
}

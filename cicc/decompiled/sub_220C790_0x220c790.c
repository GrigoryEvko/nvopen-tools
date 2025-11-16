// Function: sub_220C790
// Address: 0x220c790
//
__int64 __fastcall sub_220C790(__int64 a1, __int64 a2)
{
  char *v3; // rdi
  __int64 v4; // rax
  const char *v5; // r12
  const char *v6; // r15
  const char *v7; // r14
  char *v8; // rax
  __int64 v9; // rdx
  char v10; // r13
  size_t v11; // rax
  size_t v12; // r12
  _QWORD *v13; // r15
  size_t v14; // rax
  size_t v15; // r12
  _QWORD *v16; // rdx
  char v17; // r12
  char v18; // r14
  char *v19; // rax
  __int64 v20; // r15
  char v21; // r12
  char *v22; // rax
  __int64 v23; // rbx
  __int64 result; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  char *v27; // rsi
  size_t v28; // rax
  size_t v29; // r12
  size_t v30; // r15
  void *v31; // rax
  void *v32; // rax
  size_t v33; // rax
  __int64 v34; // rdx
  void *v35; // rax
  size_t v36; // rdx
  size_t v37; // r15
  char *v38; // rsi
  __int64 v39; // rax
  char *s; // [rsp+0h] [rbp-58h]
  char *sa; // [rsp+0h] [rbp-58h]
  size_t n; // [rsp+8h] [rbp-50h]
  size_t na; // [rsp+8h] [rbp-50h]
  size_t nb; // [rsp+8h] [rbp-50h]
  __int64 v45; // [rsp+10h] [rbp-48h]
  void *v46; // [rsp+10h] [rbp-48h]
  size_t v47; // [rsp+18h] [rbp-40h]

  if ( !*(_QWORD *)(a1 + 16) )
  {
    v39 = sub_22077B0(0x70u);
    *(_DWORD *)(v39 + 8) = 0;
    *(_QWORD *)(v39 + 16) = 0;
    *(_QWORD *)v39 = off_4A04860;
    *(_QWORD *)(v39 + 24) = 0;
    *(_WORD *)(v39 + 32) = 0;
    *(_BYTE *)(v39 + 34) = 0;
    *(_QWORD *)(v39 + 40) = 0;
    *(_QWORD *)(v39 + 48) = 0;
    *(_QWORD *)(v39 + 56) = 0;
    *(_QWORD *)(v39 + 64) = 0;
    *(_QWORD *)(v39 + 72) = 0;
    *(_QWORD *)(v39 + 80) = 0;
    *(_QWORD *)(v39 + 88) = 0;
    *(_DWORD *)(v39 + 96) = 0;
    *(_BYTE *)(v39 + 111) = 0;
    *(_QWORD *)(a1 + 16) = v39;
  }
  if ( a2 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + 33LL) = *(_BYTE *)__nl_langinfo_l();
    v3 = (char *)__nl_langinfo_l();
    if ( *v3 && v3[1] )
    {
      v25 = *(_QWORD *)(a1 + 16);
      *(_BYTE *)(v25 + 34) = sub_220E780(v3);
    }
    else
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 16) + 34LL) = *v3;
    }
    v4 = *(_QWORD *)(a1 + 16);
    if ( *(_BYTE *)(v4 + 33) )
    {
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 88LL) = *(char *)__nl_langinfo_l();
    }
    else
    {
      *(_DWORD *)(v4 + 88) = 0;
      *(_BYTE *)(v4 + 33) = 46;
    }
    v5 = (const char *)__nl_langinfo_l();
    v6 = (const char *)__nl_langinfo_l();
    s = (char *)__nl_langinfo_l();
    v7 = (const char *)__nl_langinfo_l();
    v8 = (char *)__nl_langinfo_l();
    v9 = *(_QWORD *)(a1 + 16);
    v10 = *v8;
    if ( *(_BYTE *)(v9 + 34) )
    {
      v45 = *(_QWORD *)(a1 + 16);
      v33 = strlen(v5);
      v34 = v45;
      n = v33;
      if ( v33 )
      {
        v47 = v33 + 1;
        v46 = (void *)sub_2207820(v33 + 1);
        memcpy(v46, v5, v47);
        v34 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v34 + 16) = v46;
      }
      else
      {
        *(_BYTE *)(v45 + 32) = 0;
        *(_QWORD *)(v45 + 16) = byte_3F871B3;
      }
      *(_QWORD *)(v34 + 24) = n;
      v11 = strlen(v6);
      v12 = v11;
      if ( !v11 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)(v9 + 24) = 0;
      *(_QWORD *)(v9 + 16) = byte_3F871B3;
      *(_BYTE *)(v9 + 32) = 0;
      *(_BYTE *)(v9 + 34) = 44;
      v11 = strlen(v6);
      v12 = v11;
      if ( !v11 )
      {
LABEL_10:
        v13 = *(_QWORD **)(a1 + 16);
        v13[7] = byte_3F871B3;
        goto LABEL_11;
      }
    }
    na = v11 + 1;
    v35 = (void *)sub_2207820(v11 + 1);
    v36 = na;
    nb = (size_t)v35;
    memcpy(v35, v6, v36);
    v13 = *(_QWORD **)(a1 + 16);
    v13[7] = nb;
LABEL_11:
    v13[8] = v12;
    if ( v10 )
    {
      v28 = strlen(s);
      v29 = v28;
      if ( v28 )
      {
        v37 = v28 + 1;
        v38 = s;
        sa = (char *)sub_2207820(v28 + 1);
        memcpy(sa, v38, v37);
        v13 = *(_QWORD **)(a1 + 16);
        v13[9] = sa;
      }
      else
      {
        v13[9] = byte_3F871B3;
      }
      v13[10] = v29;
      v14 = strlen(v7);
      v15 = v14;
      if ( !v14 )
        goto LABEL_13;
    }
    else
    {
      v13[10] = 2;
      v13[9] = "()";
      v14 = strlen(v7);
      v15 = v14;
      if ( !v14 )
      {
LABEL_13:
        v16 = v13;
        v13[5] = byte_3F871B3;
LABEL_14:
        v16[6] = v15;
        v17 = *(_BYTE *)__nl_langinfo_l();
        v18 = *(_BYTE *)__nl_langinfo_l();
        v19 = (char *)__nl_langinfo_l();
        v20 = *(_QWORD *)(a1 + 16);
        *(_DWORD *)(v20 + 92) = sub_220C5E0(v17, v18, *v19);
        v21 = *(_BYTE *)__nl_langinfo_l();
        v22 = (char *)__nl_langinfo_l();
        v23 = *(_QWORD *)(a1 + 16);
        result = sub_220C5E0(v21, *v22, v10);
        *(_DWORD *)(v23 + 96) = result;
        return result;
      }
    }
    v30 = v14 + 1;
    v31 = (void *)sub_2207820(v14 + 1);
    v32 = memcpy(v31, v7, v30);
    v16 = *(_QWORD **)(a1 + 16);
    v16[5] = v32;
    goto LABEL_14;
  }
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 33LL) = 46;
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 34LL) = 44;
  v26 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(v26 + 16) = byte_3F871B3;
  *(_QWORD *)(v26 + 40) = byte_3F871B3;
  *(_QWORD *)(v26 + 56) = byte_3F871B3;
  *(_QWORD *)(v26 + 72) = byte_3F871B3;
  *(_QWORD *)(v26 + 24) = 0;
  *(_BYTE *)(v26 + 32) = 0;
  *(_QWORD *)(v26 + 48) = 0;
  *(_QWORD *)(v26 + 64) = 0;
  *(_QWORD *)(v26 + 80) = 0;
  *(_DWORD *)(v26 + 88) = 0;
  *(_DWORD *)(v26 + 92) = unk_4363345;
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 96LL) = unk_4363345;
  v27 = off_4CDFAD0;
  for ( result = 0; result != 11; ++result )
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + result + 100) = v27[result];
  return result;
}

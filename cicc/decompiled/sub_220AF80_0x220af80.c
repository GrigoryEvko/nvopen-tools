// Function: sub_220AF80
// Address: 0x220af80
//
__int64 __fastcall sub_220AF80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  const char *v3; // r12
  const char *v4; // r15
  const char *v5; // r14
  char *v6; // rax
  __int64 v7; // rdx
  char v8; // r13
  size_t v9; // rax
  size_t v10; // r12
  _QWORD *v11; // r15
  size_t v12; // rax
  size_t v13; // r12
  _QWORD *v14; // rdx
  char v15; // r12
  char v16; // r14
  char *v17; // rax
  __int64 v18; // r15
  char v19; // r12
  char *v20; // rax
  __int64 v21; // rbx
  __int64 result; // rax
  __int64 v23; // rax
  char *v24; // rsi
  size_t v25; // rax
  size_t v26; // r12
  size_t v27; // r15
  void *v28; // rax
  void *v29; // rax
  size_t v30; // rax
  __int64 v31; // rdx
  void *v32; // rax
  size_t v33; // rdx
  size_t v34; // r15
  char *v35; // rsi
  __int64 v36; // rax
  char *s; // [rsp+0h] [rbp-58h]
  char *sa; // [rsp+0h] [rbp-58h]
  size_t n; // [rsp+8h] [rbp-50h]
  size_t na; // [rsp+8h] [rbp-50h]
  size_t nb; // [rsp+8h] [rbp-50h]
  __int64 v42; // [rsp+10h] [rbp-48h]
  void *v43; // [rsp+10h] [rbp-48h]
  size_t v44; // [rsp+18h] [rbp-40h]

  if ( !*(_QWORD *)(a1 + 16) )
  {
    v36 = sub_22077B0(0x70u);
    *(_DWORD *)(v36 + 8) = 0;
    *(_QWORD *)(v36 + 16) = 0;
    *(_QWORD *)v36 = off_4A04880;
    *(_QWORD *)(v36 + 24) = 0;
    *(_WORD *)(v36 + 32) = 0;
    *(_BYTE *)(v36 + 34) = 0;
    *(_QWORD *)(v36 + 40) = 0;
    *(_QWORD *)(v36 + 48) = 0;
    *(_QWORD *)(v36 + 56) = 0;
    *(_QWORD *)(v36 + 64) = 0;
    *(_QWORD *)(v36 + 72) = 0;
    *(_QWORD *)(v36 + 80) = 0;
    *(_QWORD *)(v36 + 88) = 0;
    *(_DWORD *)(v36 + 96) = 0;
    *(_BYTE *)(v36 + 111) = 0;
    *(_QWORD *)(a1 + 16) = v36;
  }
  if ( a2 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + 33LL) = *(_BYTE *)__nl_langinfo_l();
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + 34LL) = *(_BYTE *)__nl_langinfo_l();
    v2 = *(_QWORD *)(a1 + 16);
    if ( *(_BYTE *)(v2 + 33) )
    {
      *(_DWORD *)(*(_QWORD *)(a1 + 16) + 88LL) = *(char *)__nl_langinfo_l();
    }
    else
    {
      *(_DWORD *)(v2 + 88) = 0;
      *(_BYTE *)(v2 + 33) = 46;
    }
    v3 = (const char *)__nl_langinfo_l();
    v4 = (const char *)__nl_langinfo_l();
    s = (char *)__nl_langinfo_l();
    v5 = (const char *)__nl_langinfo_l();
    v6 = (char *)__nl_langinfo_l();
    v7 = *(_QWORD *)(a1 + 16);
    v8 = *v6;
    if ( *(_BYTE *)(v7 + 34) )
    {
      v42 = *(_QWORD *)(a1 + 16);
      v30 = strlen(v3);
      v31 = v42;
      n = v30;
      if ( v30 )
      {
        v44 = v30 + 1;
        v43 = (void *)sub_2207820(v30 + 1);
        memcpy(v43, v3, v44);
        v31 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(v31 + 16) = v43;
      }
      else
      {
        *(_BYTE *)(v42 + 32) = 0;
        *(_QWORD *)(v42 + 16) = byte_3F871B3;
      }
      *(_QWORD *)(v31 + 24) = n;
      v9 = strlen(v4);
      v10 = v9;
      if ( !v9 )
        goto LABEL_8;
    }
    else
    {
      *(_QWORD *)(v7 + 24) = 0;
      *(_QWORD *)(v7 + 16) = byte_3F871B3;
      *(_BYTE *)(v7 + 32) = 0;
      *(_BYTE *)(v7 + 34) = 44;
      v9 = strlen(v4);
      v10 = v9;
      if ( !v9 )
      {
LABEL_8:
        v11 = *(_QWORD **)(a1 + 16);
        v11[7] = byte_3F871B3;
        goto LABEL_9;
      }
    }
    na = v9 + 1;
    v32 = (void *)sub_2207820(v9 + 1);
    v33 = na;
    nb = (size_t)v32;
    memcpy(v32, v4, v33);
    v11 = *(_QWORD **)(a1 + 16);
    v11[7] = nb;
LABEL_9:
    v11[8] = v10;
    if ( v8 )
    {
      v25 = strlen(s);
      v26 = v25;
      if ( v25 )
      {
        v34 = v25 + 1;
        v35 = s;
        sa = (char *)sub_2207820(v25 + 1);
        memcpy(sa, v35, v34);
        v11 = *(_QWORD **)(a1 + 16);
        v11[9] = sa;
      }
      else
      {
        v11[9] = byte_3F871B3;
      }
      v11[10] = v26;
      v12 = strlen(v5);
      v13 = v12;
      if ( !v12 )
        goto LABEL_11;
    }
    else
    {
      v11[10] = 2;
      v11[9] = "()";
      v12 = strlen(v5);
      v13 = v12;
      if ( !v12 )
      {
LABEL_11:
        v14 = v11;
        v11[5] = byte_3F871B3;
LABEL_12:
        v14[6] = v13;
        v15 = *(_BYTE *)__nl_langinfo_l();
        v16 = *(_BYTE *)__nl_langinfo_l();
        v17 = (char *)__nl_langinfo_l();
        v18 = *(_QWORD *)(a1 + 16);
        *(_DWORD *)(v18 + 92) = sub_220C5E0((unsigned int)v15, (unsigned int)v16, (unsigned int)*v17);
        v19 = *(_BYTE *)__nl_langinfo_l();
        v20 = (char *)__nl_langinfo_l();
        v21 = *(_QWORD *)(a1 + 16);
        result = sub_220C5E0((unsigned int)v19, (unsigned int)*v20, (unsigned int)v8);
        *(_DWORD *)(v21 + 96) = result;
        return result;
      }
    }
    v27 = v12 + 1;
    v28 = (void *)sub_2207820(v12 + 1);
    v29 = memcpy(v28, v5, v27);
    v14 = *(_QWORD **)(a1 + 16);
    v14[5] = v29;
    goto LABEL_12;
  }
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 33LL) = 46;
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 34LL) = 44;
  v23 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(v23 + 16) = byte_3F871B3;
  *(_QWORD *)(v23 + 40) = byte_3F871B3;
  *(_QWORD *)(v23 + 56) = byte_3F871B3;
  *(_QWORD *)(v23 + 72) = byte_3F871B3;
  *(_QWORD *)(v23 + 24) = 0;
  *(_BYTE *)(v23 + 32) = 0;
  *(_QWORD *)(v23 + 48) = 0;
  *(_QWORD *)(v23 + 64) = 0;
  *(_QWORD *)(v23 + 80) = 0;
  *(_DWORD *)(v23 + 88) = 0;
  *(_DWORD *)(v23 + 92) = unk_4363345;
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 96LL) = unk_4363345;
  v24 = off_4CDFAD0;
  for ( result = 0; result != 11; ++result )
    *(_BYTE *)(*(_QWORD *)(a1 + 16) + result + 100) = v24[result];
  return result;
}

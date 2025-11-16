// Function: sub_941230
// Address: 0x941230
//
int __fastcall sub_941230(__int64 a1, unsigned int a2, __int16 a3)
{
  const char **v5; // rax
  const char ***v6; // rsi
  int v7; // r9d
  __int64 v8; // rdx
  const char *v9; // rbx
  const char **v10; // r14
  size_t v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  __int64 *v15; // r15
  __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rdx
  unsigned __int8 v19; // al
  __int64 v20; // r14
  const char **v21; // rsi
  __int64 v22; // rdi
  unsigned __int8 v23; // dl
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 *v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v33[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( a2 && a3 && *(_DWORD *)(a1 + 448) != a2 )
  {
    *(_DWORD *)(a1 + 448) = a2;
    *(_WORD *)(a1 + 452) = a3;
  }
  v5 = *(const char ***)(a1 + 512);
  if ( *(const char ***)(a1 + 480) == v5 )
    return (int)v5;
  v6 = 0;
  v5 = (const char **)sub_93ED80(a2, 0);
  v8 = *(_QWORD *)(a1 + 512);
  if ( v8 == *(_QWORD *)(a1 + 520) )
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 536) - 8LL) + 512LL;
  if ( !v5 )
    return (int)v5;
  v9 = *v5;
  v10 = *(const char ***)(v8 - 8);
  v11 = 0;
  if ( *v5 )
    v11 = strlen(*v5);
  v12 = *(unsigned __int8 *)v10;
  v5 = v10;
  v13 = v12;
  if ( (_BYTE)v12 == 16
    || ((v14 = *((_BYTE *)v10 - 16), (v14 & 2) != 0)
      ? (v6 = (const char ***)*(v10 - 4))
      : (v6 = (const char ***)&v10[-((v14 >> 2) & 0xF) - 2]),
        (v5 = *v6) != 0) )
  {
    v23 = *((_BYTE *)v5 - 16);
    if ( (v23 & 2) != 0 )
      v5 = (const char **)*(v5 - 4);
    else
      v5 = &v5[-((v23 >> 2) & 0xF) - 2];
    if ( *v5 )
    {
      v5 = (const char **)sub_B91420(*v5, v6);
      if ( v11 == v24 )
      {
        if ( v11 )
        {
          LODWORD(v5) = memcmp(v5, v9, v11);
          if ( (_DWORD)v5 )
          {
            v13 = *(unsigned __int8 *)v10;
            v15 = (__int64 *)(a1 + 464);
            if ( (_BYTE)v13 == 20 )
              goto LABEL_17;
            goto LABEL_32;
          }
        }
        return (int)v5;
      }
      v12 = *(unsigned __int8 *)v10;
    }
    else if ( !v11 )
    {
      return (int)v5;
    }
    v13 = (unsigned int)v12;
LABEL_16:
    v15 = (__int64 *)(a1 + 464);
    if ( (_BYTE)v13 == 20 )
    {
LABEL_17:
      v16 = *(_QWORD *)(a1 + 512);
      if ( v16 == *(_QWORD *)(a1 + 520) )
      {
        j_j___libc_free_0(v16, 512);
        v28 = (__int64 *)(*(_QWORD *)(a1 + 536) - 8LL);
        *(_QWORD *)(a1 + 536) = v28;
        v29 = *v28;
        v13 = *v28 + 512;
        *(_QWORD *)(a1 + 520) = v29;
        *(_QWORD *)(a1 + 528) = v13;
        *(_QWORD *)(a1 + 512) = v29 + 504;
        if ( *(_QWORD *)(v29 + 504) )
          sub_B91220(v29 + 504);
      }
      else
      {
        v17 = v16 - 8;
        *(_QWORD *)(a1 + 512) = v16 - 8;
        if ( *(_QWORD *)(v16 - 8) )
          sub_B91220(v16 - 8);
      }
      v18 = sub_9405D0(a1, a2, v13, v12, v17, v7);
      v19 = *((_BYTE *)v10 - 16);
      if ( (v19 & 2) != 0 )
        v20 = (__int64)*(v10 - 4);
      else
        v20 = (__int64)&v10[-((v19 >> 2) & 0xF) - 2];
      v21 = *(const char ***)(v20 + 8);
      v22 = a1 + 16;
LABEL_23:
      v33[0] = sub_ADD750(v22, v21, v18, 0);
      LODWORD(v5) = sub_940130(v15, v33);
      return (int)v5;
    }
LABEL_32:
    v25 = (unsigned int)(v13 - 18);
    if ( (unsigned __int8)v25 > 1u )
      return (int)v5;
    v26 = *(_QWORD *)(a1 + 512);
    if ( v26 == *(_QWORD *)(a1 + 520) )
    {
      j_j___libc_free_0(v26, 512);
      v30 = (__int64 *)(*(_QWORD *)(a1 + 536) - 8LL);
      *(_QWORD *)(a1 + 536) = v30;
      v31 = *v30;
      v25 = *v30 + 512;
      *(_QWORD *)(a1 + 520) = v31;
      *(_QWORD *)(a1 + 528) = v25;
      *(_QWORD *)(a1 + 512) = v31 + 504;
      if ( *(_QWORD *)(v31 + 504) )
        sub_B91220(v31 + 504);
    }
    else
    {
      v27 = v26 - 8;
      *(_QWORD *)(a1 + 512) = v26 - 8;
      if ( *(_QWORD *)(v26 - 8) )
        sub_B91220(v26 - 8);
    }
    v22 = a1 + 16;
    v21 = v10;
    v18 = sub_9405D0(a1, a2, v25, v12, v27, v7);
    goto LABEL_23;
  }
  if ( v11 )
    goto LABEL_16;
  return (int)v5;
}

// Function: sub_BCB4B0
// Address: 0xbcb4b0
//
int __fastcall sub_BCB4B0(__int64 **a1, const void *a2, size_t a3)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rsi
  __int64 v9; // r14
  __int64 *v10; // rdi
  __int64 *v11; // rsi
  unsigned int v12; // eax
  unsigned int v13; // r8d
  size_t **v14; // r9
  _BYTE *v15; // rdi
  size_t v16; // rax
  unsigned __int64 v17; // r11
  size_t v18; // rax
  _BYTE *v19; // rax
  _BYTE *i; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r15
  size_t v24; // r12
  unsigned int v25; // eax
  unsigned int v26; // r9d
  _QWORD *v27; // rcx
  __int64 v28; // rax
  unsigned int v29; // r9d
  _QWORD *v30; // rcx
  _QWORD *v31; // r11
  __int64 v32; // rsi
  __int64 *v33; // r12
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rdi
  __int64 *v37; // rax
  __int64 v38; // rdx
  const void *v39; // rsi
  _QWORD *v41; // [rsp+8h] [rbp-108h]
  size_t **v42; // [rsp+8h] [rbp-108h]
  _QWORD *v43; // [rsp+8h] [rbp-108h]
  unsigned __int64 v44; // [rsp+10h] [rbp-100h]
  unsigned int v45; // [rsp+10h] [rbp-100h]
  unsigned int v46; // [rsp+10h] [rbp-100h]
  _QWORD *v47; // [rsp+10h] [rbp-100h]
  void *src; // [rsp+18h] [rbp-F8h]
  size_t *srcc; // [rsp+18h] [rbp-F8h]
  unsigned int srcb; // [rsp+18h] [rbp-F8h]
  void *srca; // [rsp+18h] [rbp-F8h]
  void *v52; // [rsp+20h] [rbp-F0h]
  _QWORD v53[6]; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE **v54; // [rsp+70h] [rbp-A0h]
  _BYTE *v55; // [rsp+80h] [rbp-90h] BYREF
  size_t v56; // [rsp+88h] [rbp-88h]
  unsigned __int64 v57; // [rsp+90h] [rbp-80h]
  _BYTE dest[120]; // [rsp+98h] [rbp-78h] BYREF

  v6 = (__int64 *)sub_BCB490((__int64)a1);
  if ( a3 == v7 )
  {
    if ( !a3 )
      return (int)v6;
    LODWORD(v6) = memcmp(a2, v6, a3);
    if ( !(_DWORD)v6 )
      return (int)v6;
    v6 = *a1;
    v11 = a1[3];
    v9 = **a1;
    if ( v11 )
    {
      sub_C929B0(v9 + 2968, v11);
LABEL_11:
      v9 = **a1;
      goto LABEL_12;
    }
  }
  else
  {
    v6 = *a1;
    v8 = a1[3];
    v9 = **a1;
    if ( v8 )
    {
      LODWORD(v6) = sub_C929B0(v9 + 2968, v8);
      if ( !a3 )
      {
        v10 = a1[3];
        if ( v10 )
        {
          LODWORD(v6) = sub_C7D6A0(v10, *v10 + 17, 8);
          a1[3] = 0;
        }
        return (int)v6;
      }
      goto LABEL_11;
    }
  }
  if ( !a3 )
    return (int)v6;
LABEL_12:
  v55 = a2;
  v56 = a3;
  v12 = sub_C92610(a2, a3);
  v13 = sub_C92740(v9 + 2968, a2, a3, v12);
  v14 = (size_t **)(*(_QWORD *)(v9 + 2968) + 8LL * v13);
  if ( *v14 )
  {
    if ( *v14 != (size_t *)-8LL )
    {
      v15 = dest;
      v56 = 0;
      v55 = dest;
      v57 = 64;
      if ( a3 > 0x40 )
      {
        sub_C8D290(&v55, dest, a3, 1);
        v15 = &v55[v56];
      }
      memcpy(v15, a2, a3);
      v16 = a3 + v56;
      v56 = v16;
      if ( v16 + 1 > v57 )
      {
        sub_C8D290(&v55, dest, v16 + 1, 1);
        v16 = v56;
      }
      v55[v16] = 46;
      v53[5] = 0x100000000LL;
      ++v56;
      v53[1] = 2;
      v53[0] = &unk_49DD288;
      memset(&v53[2], 0, 24);
      v54 = &v55;
      sub_CB5980(v53, 0, 0, 0);
      v17 = (unsigned int)(a3 + 1);
      while ( 1 )
      {
        v18 = v56;
        if ( v17 != v56 )
        {
          if ( v17 >= v56 )
          {
            if ( v17 > v57 )
            {
              srca = (void *)v17;
              sub_C8D290(&v55, dest, v17, 1);
              v18 = v56;
              v17 = (unsigned __int64)srca;
            }
            v19 = &v55[v18];
            for ( i = &v55[v17]; i != v19; ++v19 )
            {
              if ( v19 )
                *v19 = 0;
            }
          }
          v56 = v17;
        }
        v44 = v17;
        v21 = **a1;
        v22 = *(unsigned int *)(v21 + 2992);
        *(_DWORD *)(v21 + 2992) = v22 + 1;
        sub_CB59D0(v53, v22);
        v23 = **a1;
        v24 = (size_t)v54[1];
        v52 = *v54;
        src = *v54;
        v25 = sub_C92610(*v54, v24);
        v26 = sub_C92740(v23 + 2968, v52, v24, v25);
        v27 = (_QWORD *)(*(_QWORD *)(v23 + 2968) + 8LL * v26);
        if ( !*v27 )
          break;
        v17 = v44;
        if ( *v27 == -8 )
        {
          --*(_DWORD *)(v23 + 2984);
          break;
        }
      }
      v41 = v27;
      v45 = v26;
      v28 = sub_C7D670(v24 + 17, 8);
      v29 = v45;
      v30 = v41;
      v31 = (_QWORD *)v28;
      if ( v24 )
      {
        v39 = src;
        v47 = v41;
        srcb = v29;
        v43 = (_QWORD *)v28;
        memcpy((void *)(v28 + 16), v39, v24);
        v29 = srcb;
        v30 = v47;
        v31 = v43;
      }
      *((_BYTE *)v31 + v24 + 16) = 0;
      v32 = v29;
      *v31 = v24;
      v31[1] = a1;
      *v30 = v31;
      ++*(_DWORD *)(v23 + 2980);
      v33 = (__int64 *)(*(_QWORD *)(v23 + 2968) + 8LL * (unsigned int)sub_C929D0(v23 + 2968, v29));
      if ( *v33 == -8 || !*v33 )
      {
        v34 = v33 + 1;
        do
        {
          do
          {
            v35 = *v34;
            v33 = v34++;
          }
          while ( !v35 );
        }
        while ( v35 == -8 );
      }
      v53[0] = &unk_49DD388;
      sub_CB5840(v53);
      if ( v55 != dest )
        _libc_free(v55, v32);
      goto LABEL_40;
    }
    --*(_DWORD *)(v9 + 2984);
  }
  v42 = v14;
  v46 = v13;
  srcc = (size_t *)sub_C7D670(a3 + 17, 8);
  memcpy(srcc + 2, a2, a3);
  *((_BYTE *)srcc + a3 + 16) = 0;
  *srcc = a3;
  srcc[1] = (size_t)a1;
  *v42 = srcc;
  ++*(_DWORD *)(v9 + 2980);
  v33 = (__int64 *)(*(_QWORD *)(v9 + 2968) + 8LL * (unsigned int)sub_C929D0(v9 + 2968, v46));
  if ( !*v33 || *v33 == -8 )
  {
    v37 = v33 + 1;
    do
    {
      do
      {
        v38 = *v37;
        v33 = v37++;
      }
      while ( !v38 );
    }
    while ( v38 == -8 );
  }
LABEL_40:
  v36 = a1[3];
  if ( v36 )
    sub_C7D6A0(v36, *v36 + 17, 8);
  v6 = (__int64 *)*v33;
  a1[3] = (__int64 *)*v33;
  return (int)v6;
}

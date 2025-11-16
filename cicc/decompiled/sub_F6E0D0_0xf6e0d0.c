// Function: sub_F6E0D0
// Address: 0xf6e0d0
//
__int64 __fastcall sub_F6E0D0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, char a5)
{
  unsigned __int8 v5; // dl
  __int64 v6; // rbx
  __int64 v7; // r13
  bool v8; // zf
  __int64 v9; // rdi
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  const char *v13; // r10
  __int64 *v14; // r12
  const char *v15; // r14
  char v16; // bl
  __int64 *v17; // r13
  __int64 v18; // r15
  unsigned __int8 v19; // al
  __int64 *v20; // rax
  size_t v21; // rdx
  size_t v22; // rax
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // r14
  char v26; // r13
  _BYTE *v27; // rax
  unsigned __int8 v28; // dl
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 *v31; // r13
  __int64 *v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // r10
  char v36; // r13
  __int64 *v38; // rdi
  __m128i *v39; // rbx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // [rsp+10h] [rbp-E0h]
  __int64 v43; // [rsp+18h] [rbp-D8h]
  void *s1; // [rsp+28h] [rbp-C8h]
  size_t v46; // [rsp+38h] [rbp-B8h]
  __int64 v48; // [rsp+40h] [rbp-B0h]
  bool v49; // [rsp+48h] [rbp-A8h]
  __int64 v50; // [rsp+48h] [rbp-A8h]
  __int64 v51; // [rsp+60h] [rbp-90h]
  __int64 *v52; // [rsp+70h] [rbp-80h] BYREF
  __int64 v53; // [rsp+78h] [rbp-78h]
  _QWORD v54[14]; // [rsp+80h] [rbp-70h] BYREF

  if ( !a1 )
  {
    if ( a5 )
      return 0;
    return v51;
  }
  v5 = *(_BYTE *)(a1 - 16);
  v6 = a1;
  v7 = a2;
  if ( !a4 )
  {
    v52 = v54;
    v53 = 0x800000001LL;
    v54[0] = 0;
LABEL_4:
    if ( (v5 & 2) != 0 )
    {
      v9 = *(_QWORD *)(a1 - 32);
      a2 = *(unsigned int *)(v6 - 24);
    }
    else
    {
      a2 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
      v9 = a1 - 8LL * ((v5 >> 2) & 0xF) - 16;
    }
    v49 = 0;
    v14 = (__int64 *)sub_F6BB50(v9, a2, 1);
    if ( v14 != v10 )
    {
      v43 = v6;
      v15 = v13;
      v16 = v11;
      v42 = v7;
      v17 = v10;
      while ( 1 )
      {
        while ( !v16 )
        {
          ++v14;
          v49 = 1;
          if ( v17 == v14 )
          {
LABEL_20:
            v6 = v43;
            v7 = v42;
            goto LABEL_24;
          }
        }
        v18 = *v14;
        v19 = *(_BYTE *)(*v14 - 16);
        if ( (v19 & 2) != 0 )
        {
          if ( !*(_DWORD *)(v18 - 24) )
            goto LABEL_54;
          v20 = *(__int64 **)(v18 - 32);
        }
        else
        {
          if ( (*(_WORD *)(v18 - 16) & 0x3C0) == 0 )
            goto LABEL_54;
          v20 = (__int64 *)(v18 - 8LL * ((v19 >> 2) & 0xF) - 16);
        }
        if ( *(_BYTE *)*v20
          || (s1 = (void *)sub_B91420(*v20), v46 = v21, v15)
          && ((v22 = strlen(v15), v22 > v46) || v22 && (a2 = (__int64)v15, memcmp(s1, v15, v22))) )
        {
LABEL_54:
          v40 = (unsigned int)v53;
          v41 = (unsigned int)v53 + 1LL;
          if ( v41 > HIDWORD(v53) )
          {
            a2 = (__int64)v54;
            sub_C8D5F0((__int64)&v52, v54, v41, 8u, v11, v12);
            v40 = (unsigned int)v53;
          }
          v52[v40] = v18;
          LODWORD(v53) = v53 + 1;
          goto LABEL_17;
        }
        v49 = v16;
LABEL_17:
        if ( v17 == ++v14 )
          goto LABEL_20;
      }
    }
    goto LABEL_24;
  }
  v8 = *a4 == 0;
  v52 = v54;
  v53 = 0x800000001LL;
  v54[0] = 0;
  if ( !v8 )
    goto LABEL_4;
  if ( (v5 & 2) != 0 )
    v23 = *(_DWORD *)(a1 - 24);
  else
    v23 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v49 = v23 > 1;
LABEL_24:
  v24 = v7 + 16 * a3;
  if ( v7 == v24 )
  {
    v36 = a5;
  }
  else
  {
    v25 = v7;
    v26 = 0;
    do
    {
      while ( 1 )
      {
        a2 = *(_QWORD *)v25;
        v27 = sub_D49670(v6, *(const void **)v25, *(_QWORD *)(v25 + 8));
        if ( v27 )
          break;
        v25 += 16;
        if ( v24 == v25 )
          goto LABEL_37;
      }
      v28 = *(v27 - 16);
      if ( (v28 & 2) != 0 )
      {
        v29 = *((_QWORD *)v27 - 4);
        a2 = *((unsigned int *)v27 - 6);
      }
      else
      {
        a2 = (*((_WORD *)v27 - 8) >> 6) & 0xF;
        v29 = (__int64)&v27[-16 - 8LL * ((v28 >> 2) & 0xF)];
      }
      v31 = (__int64 *)sub_F6BB50(v29, a2, 1);
      v33 = (__int64)v32;
      if ( v32 != v31 )
      {
        v34 = (unsigned int)v53;
        do
        {
          v35 = *v31;
          if ( v34 + 1 > (unsigned __int64)HIDWORD(v53) )
          {
            a2 = (__int64)v54;
            v48 = v33;
            v50 = *v31;
            sub_C8D5F0((__int64)&v52, v54, v34 + 1, 8u, v30, v33);
            v34 = (unsigned int)v53;
            v33 = v48;
            v35 = v50;
          }
          ++v31;
          v52[v34] = v35;
          v34 = (unsigned int)(v53 + 1);
          LODWORD(v53) = v53 + 1;
        }
        while ( (__int64 *)v33 != v31 );
        v49 = 1;
      }
      v25 += 16;
      v26 = 1;
    }
    while ( v24 != v25 );
LABEL_37:
    v36 = a5 | v26;
  }
  if ( v36 )
  {
    if ( v49 || a5 )
    {
      if ( (unsigned int)v53 == 1 )
      {
        v51 = 0;
      }
      else
      {
        v38 = (__int64 *)(*(_QWORD *)(v6 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v6 + 8) & 4) != 0 )
          v38 = (__int64 *)*v38;
        a2 = 0;
        v39 = (__m128i *)sub_B9C770(v38, v52, (__int64 *)(unsigned int)v53, 0, 1);
        sub_BA6610(v39, 0, (unsigned __int8 *)v39);
        v51 = (__int64)v39;
      }
    }
    else
    {
      v51 = v6;
    }
  }
  if ( v52 != v54 )
    _libc_free(v52, a2);
  return v51;
}

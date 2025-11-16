// Function: sub_2D54BE0
// Address: 0x2d54be0
//
__int64 __fastcall sub_2D54BE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // rax
  _QWORD **v10; // r14
  _QWORD *v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // r12
  __int64 p_src; // rcx
  __int64 v15; // rbx
  void *v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int8 v19; // dl
  __int64 v20; // rax
  _BYTE *v21; // rax
  unsigned __int8 v22; // dl
  unsigned __int8 v23; // dl
  char **v24; // rax
  char *v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  char *v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r9
  size_t v32; // rdi
  size_t v33; // r15
  const char *v34; // r13
  size_t v35; // rdx
  size_t v36; // r15
  int v37; // eax
  _QWORD *v38; // r10
  __int64 v40; // rax
  __int64 v41; // r9
  _QWORD *v42; // r10
  _QWORD *v43; // r8
  size_t v44; // r13
  void *v45; // rax
  void *v46; // rax
  size_t v47; // rdx
  _QWORD *v48; // [rsp+10h] [rbp-110h]
  _QWORD *v49; // [rsp+10h] [rbp-110h]
  _QWORD *v50; // [rsp+10h] [rbp-110h]
  _QWORD *v51; // [rsp+18h] [rbp-108h]
  _QWORD *v52; // [rsp+18h] [rbp-108h]
  int v53; // [rsp+18h] [rbp-108h]
  unsigned int v54; // [rsp+20h] [rbp-100h]
  char *v55; // [rsp+20h] [rbp-100h]
  int v56; // [rsp+20h] [rbp-100h]
  _QWORD *v57; // [rsp+20h] [rbp-100h]
  __int64 *v58; // [rsp+28h] [rbp-F8h]
  __int64 v59; // [rsp+30h] [rbp-F0h]
  __int64 v60; // [rsp+38h] [rbp-E8h]
  __int64 v61; // [rsp+48h] [rbp-D8h] BYREF
  void *src; // [rsp+50h] [rbp-D0h] BYREF
  size_t n; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v64; // [rsp+60h] [rbp-C0h]
  _BYTE v65[184]; // [rsp+68h] [rbp-B8h] BYREF

  if ( !*(_QWORD *)(a1 + 176) )
    return 0;
  v6 = a1;
  v58 = (__int64 *)(a1 + 248);
  if ( *(_DWORD *)(a1 + 260) )
  {
    v7 = 0;
    v8 = *(unsigned int *)(a1 + 256);
    if ( (_DWORD)v8 )
    {
      v60 = 8 * v8;
      do
      {
        v10 = (_QWORD **)(v7 + *(_QWORD *)(a1 + 248));
        v11 = *v10;
        if ( *v10 != (_QWORD *)-8LL && v11 )
        {
          v12 = v11[1];
          v13 = *v11 + 161LL;
          if ( (_QWORD *)v12 != v11 + 4 )
            _libc_free(v12);
          sub_C7D6A0((__int64)v11, v13, 8);
        }
        *v10 = 0;
        v7 += 8;
      }
      while ( v7 != v60 );
      v6 = a1;
    }
    *(_QWORD *)(v6 + 260) = 0;
  }
  p_src = (__int64)&src;
  v15 = *(_QWORD *)(a2 + 32);
  v59 = a2 + 24;
  if ( a2 + 24 != v15 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = v15 - 56;
        n = 0;
        v64 = 128;
        if ( !v15 )
          v17 = 0;
        src = v65;
        if ( !sub_B2FC80(v17) )
          break;
        v16 = src;
        if ( src != v65 )
          goto LABEL_16;
LABEL_17:
        v15 = *(_QWORD *)(v15 + 8);
        if ( v59 == v15 )
          goto LABEL_41;
      }
      v18 = sub_B92180(v17);
      if ( v18 )
      {
        v19 = *(_BYTE *)(v18 - 16);
        v20 = (v19 & 2) != 0 ? *(_QWORD *)(v18 - 32) : v18 - 16 - 8LL * ((v19 >> 2) & 0xF);
        v21 = *(_BYTE **)(v20 + 40);
        if ( v21 )
          break;
      }
LABEL_37:
      v34 = sub_BD5D20(v17);
      v36 = v35;
      v37 = sub_C92610();
      p_src = (unsigned int)sub_C92740((__int64)v58, v34, v36, v37);
      a6 = p_src;
      v38 = (_QWORD *)(*(_QWORD *)(v6 + 248) + 8 * p_src);
      if ( *v38 )
      {
        if ( *v38 != -8 )
          goto LABEL_39;
        --*(_DWORD *)(v6 + 264);
      }
      v51 = v38;
      v54 = p_src;
      v40 = sub_C7D670(v36 + 161, 8);
      v41 = v54;
      v42 = v51;
      v43 = (_QWORD *)v40;
      if ( v36 )
      {
        v48 = (_QWORD *)v40;
        memcpy((void *)(v40 + 160), v34, v36);
        v41 = v54;
        v42 = v51;
        v43 = v48;
      }
      v44 = n;
      v45 = v43 + 4;
      *((_BYTE *)v43 + v36 + 160) = 0;
      *v43 = v36;
      v43[1] = v43 + 4;
      v43[2] = 0;
      v43[3] = 128;
      if ( v44 )
      {
        v47 = v44;
        if ( v44 <= 0x80 )
          goto LABEL_54;
        v50 = v42;
        v53 = v41;
        v57 = v43;
        sub_C8D290((__int64)(v43 + 1), v43 + 4, v44, 1u, (__int64)v43, v41);
        v47 = n;
        v43 = v57;
        LODWORD(v41) = v53;
        v42 = v50;
        v45 = (void *)v57[1];
        if ( n )
        {
LABEL_54:
          v49 = v43;
          v52 = v42;
          v56 = v41;
          memcpy(v45, src, v47);
          v43 = v49;
          v42 = v52;
          LODWORD(v41) = v56;
        }
        v43[2] = v44;
      }
      *v42 = v43;
      ++*(_DWORD *)(v6 + 260);
      sub_C929D0(v58, v41);
LABEL_39:
      v16 = src;
      if ( src != v65 )
      {
LABEL_16:
        _libc_free((unsigned __int64)v16);
        goto LABEL_17;
      }
      v15 = *(_QWORD *)(v15 + 8);
      if ( v59 == v15 )
        goto LABEL_41;
    }
    if ( *v21 == 16 )
      goto LABEL_28;
    v22 = *(v21 - 16);
    if ( (v22 & 2) != 0 )
    {
      v21 = (_BYTE *)**((_QWORD **)v21 - 4);
      if ( v21 )
        goto LABEL_28;
    }
    else
    {
      v21 = *(_BYTE **)&v21[-8 * ((v22 >> 2) & 0xF) - 16];
      if ( v21 )
      {
LABEL_28:
        v23 = *(v21 - 16);
        if ( (v23 & 2) != 0 )
          v24 = (char **)*((_QWORD *)v21 - 4);
        else
          v24 = (char **)&v21[-8 * ((v23 >> 2) & 0xF) - 16];
        v25 = *v24;
        if ( *v24 )
        {
          v25 = (char *)sub_B91420((__int64)v25);
          v27 = v26;
        }
        else
        {
          v27 = 0;
        }
        goto LABEL_32;
      }
    }
    v27 = 0;
    v25 = (char *)byte_3F871B3;
LABEL_32:
    v28 = sub_C81F40(v25, v27, 0);
    n = 0;
    v32 = 0;
    v33 = v29;
    if ( v29 > v64 )
    {
      v55 = v28;
      sub_C8D290((__int64)&src, v65, v29, 1u, v30, v31);
      v32 = n;
      v28 = v55;
    }
    if ( v33 )
    {
      memcpy((char *)src + v32, v28, v33);
      v32 = n;
    }
    n = v33 + v32;
    goto LABEL_37;
  }
LABEL_41:
  sub_2D54A80(&v61, (__int64 *)(v6 + 176), a3, p_src, a5, a6);
  if ( (v61 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v46 = (void *)(v61 & 0xFFFFFFFFFFFFFFFELL | 1);
    v61 = 0;
    src = v46;
    sub_C641D0((__int64 *)&src, 1u);
  }
  return 0;
}

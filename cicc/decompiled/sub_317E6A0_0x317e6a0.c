// Function: sub_317E6A0
// Address: 0x317e6a0
//
__int64 __fastcall sub_317E6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int8 v4; // al
  bool v5; // dl
  unsigned __int8 **v6; // rdx
  unsigned __int8 *v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  const __m128i *v16; // rdx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r8
  __m128i *v19; // rax
  __int64 v20; // r12
  int v21; // eax
  __int64 v22; // r13
  unsigned __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v26; // rax
  __int64 v27; // r13
  unsigned __int8 **i; // rdx
  unsigned __int8 *v29; // rax
  unsigned __int8 v30; // dl
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // r12
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  const __m128i *v43; // rdx
  __m128i *v44; // rax
  __int64 v45; // rdx
  bool v46; // dl
  __int64 v47; // rax
  unsigned __int8 v48; // al
  unsigned __int8 **v49; // r12
  unsigned __int8 *v50; // rax
  unsigned __int8 v51; // dl
  __int64 v52; // rdx
  __int64 v53; // rdi
  unsigned __int64 v54; // r12
  unsigned __int8 v55; // al
  unsigned __int8 **v56; // r12
  unsigned __int8 *v57; // rax
  unsigned __int8 v58; // dl
  __int64 v59; // rax
  __int64 v60; // rax
  char *v61; // rbx
  __int64 v63; // [rsp+10h] [rbp-150h] BYREF
  __int64 v64; // [rsp+18h] [rbp-148h]
  __int64 v65; // [rsp+20h] [rbp-140h]
  unsigned __int64 v66; // [rsp+30h] [rbp-130h] BYREF
  __int64 v67; // [rsp+38h] [rbp-128h]
  _BYTE v68[288]; // [rsp+40h] [rbp-120h] BYREF

  v2 = a2 - 16;
  v3 = a2;
  v66 = (unsigned __int64)v68;
  v67 = 0xA00000000LL;
  v4 = *(_BYTE *)(a2 - 16);
  v5 = (v4 & 2) != 0;
  if ( (v4 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) != 2 )
      goto LABEL_3;
    v26 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( !v26 )
      goto LABEL_3;
  }
  else if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 || (v26 = *(_QWORD *)(v2 - 8LL * ((v4 >> 2) & 0xF) + 8)) == 0 )
  {
LABEL_44:
    v6 = (unsigned __int8 **)(v2 - 8LL * ((*(_BYTE *)(v3 - 16) >> 2) & 0xF));
    goto LABEL_4;
  }
  v27 = a2;
  v3 = v26;
  if ( !v5 )
    goto LABEL_33;
LABEL_20:
  for ( i = *(unsigned __int8 ***)(v27 - 32); ; i = (unsigned __int8 **)(v2 - 8LL * ((*(_BYTE *)(v27 - 16) >> 2) & 0xF)) )
  {
    v29 = sub_AF34D0(*i);
    v30 = *(v29 - 16);
    if ( (v30 & 2) != 0 )
    {
      v31 = *(_QWORD *)(*((_QWORD *)v29 - 4) + 24LL);
      if ( !v31 )
        goto LABEL_35;
    }
    else
    {
      v31 = *(_QWORD *)&v29[-8 * ((v30 >> 2) & 0xF) + 8];
      if ( !v31 )
        goto LABEL_35;
    }
    v32 = sub_B91420(v31);
    v34 = v33;
    if ( v33 )
    {
      v35 = v32;
      v36 = v33;
      goto LABEL_25;
    }
LABEL_35:
    v48 = *(_BYTE *)(v27 - 16);
    v49 = (v48 & 2) != 0 ? *(unsigned __int8 ***)(v27 - 32) : (unsigned __int8 **)(v2 - 8LL * ((v48 >> 2) & 0xF));
    v50 = sub_AF34D0(*v49);
    v51 = *(v50 - 16);
    if ( (v51 & 2) != 0 )
    {
      v35 = *(_QWORD *)(*((_QWORD *)v50 - 4) + 16LL);
      if ( !v35 )
        goto LABEL_46;
    }
    else
    {
      v35 = *(_QWORD *)&v50[-8 * ((v51 >> 2) & 0xF)];
      if ( !v35 )
      {
LABEL_46:
        v34 = 0;
        goto LABEL_27;
      }
    }
    v35 = sub_B91420(v35);
    v34 = v52;
    v36 = v52;
    if ( !v52 )
      goto LABEL_27;
LABEL_25:
    if ( unk_4F838D1 )
    {
      v53 = v35;
      v35 = 0;
      v34 = sub_B2F650(v53, v36);
    }
LABEL_27:
    v37 = sub_C1B090(v3, 0);
    v64 = v35;
    v63 = v37;
    v40 = (unsigned int)v67;
    v65 = v34;
    v41 = (unsigned int)v67 + 1LL;
    if ( v41 > HIDWORD(v67) )
    {
      v54 = v66;
      if ( v66 > (unsigned __int64)&v63 || (unsigned __int64)&v63 >= v66 + 24LL * (unsigned int)v67 )
      {
        sub_C8D5F0((__int64)&v66, v68, v41, 0x18u, v38, v39);
        v42 = v66;
        v40 = (unsigned int)v67;
        v43 = (const __m128i *)&v63;
      }
      else
      {
        sub_C8D5F0((__int64)&v66, v68, v41, 0x18u, v38, v39);
        v42 = v66;
        v40 = (unsigned int)v67;
        v43 = (const __m128i *)((char *)&v63 + v66 - v54);
      }
    }
    else
    {
      v42 = v66;
      v43 = (const __m128i *)&v63;
    }
    v2 = v3 - 16;
    v44 = (__m128i *)(v42 + 24 * v40);
    *v44 = _mm_loadu_si128(v43);
    v45 = v43[1].m128i_i64[0];
    LODWORD(v67) = v67 + 1;
    v44[1].m128i_i64[0] = v45;
    v46 = (*(_BYTE *)(v3 - 16) & 2) != 0;
    if ( (*(_BYTE *)(v3 - 16) & 2) == 0 )
      break;
    if ( *(_DWORD *)(v3 - 24) != 2 )
      goto LABEL_3;
    v47 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 8LL);
    v27 = v3;
    if ( !v47 )
      goto LABEL_43;
LABEL_32:
    v3 = v47;
    if ( v46 )
      goto LABEL_20;
LABEL_33:
    ;
  }
  if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xF) != 2 )
    goto LABEL_44;
  v27 = v3;
  v47 = *(_QWORD *)(v2 - 8LL * ((*(_BYTE *)(v3 - 16) >> 2) & 0xF) + 8);
  if ( v47 )
    goto LABEL_32;
LABEL_43:
  if ( (*(_BYTE *)(v3 - 16) & 2) == 0 )
    goto LABEL_44;
LABEL_3:
  v6 = *(unsigned __int8 ***)(v3 - 32);
LABEL_4:
  v7 = sub_AF34D0(*v6);
  v8 = *(v7 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(_QWORD *)(*((_QWORD *)v7 - 4) + 24LL);
    if ( v9 )
      goto LABEL_6;
LABEL_53:
    v55 = *(_BYTE *)(v3 - 16);
    if ( (v55 & 2) != 0 )
      v56 = *(unsigned __int8 ***)(v3 - 32);
    else
      v56 = (unsigned __int8 **)(v2 - 8LL * ((v55 >> 2) & 0xF));
    v57 = sub_AF34D0(*v56);
    v58 = *(v57 - 16);
    if ( (v58 & 2) != 0 )
      v59 = *((_QWORD *)v57 - 4);
    else
      v59 = (__int64)&v57[-8 * ((v58 >> 2) & 0xF) - 16];
    v13 = *(_QWORD *)(v59 + 16);
    if ( v13 )
    {
      v13 = sub_B91420(v13);
      v14 = v11;
      if ( v11 )
        goto LABEL_8;
    }
    else
    {
      v11 = 0;
    }
  }
  else
  {
    v9 = *(_QWORD *)&v7[-8 * ((v8 >> 2) & 0xF) + 8];
    if ( !v9 )
      goto LABEL_53;
LABEL_6:
    v10 = sub_B91420(v9);
    if ( !v11 )
      goto LABEL_53;
    v13 = v10;
    v14 = v11;
LABEL_8:
    if ( unk_4F838D1 )
    {
      v60 = sub_B2F650(v13, v14);
      v13 = 0;
      v11 = v60;
    }
  }
  v15 = (unsigned int)v67;
  v65 = v11;
  v16 = (const __m128i *)&v63;
  v64 = v13;
  v17 = v66;
  v63 = 0;
  v18 = (unsigned int)v67 + 1LL;
  if ( v18 > HIDWORD(v67) )
  {
    if ( v66 > (unsigned __int64)&v63 || (unsigned __int64)&v63 >= v66 + 24LL * (unsigned int)v67 )
    {
      sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 0x18u, v18, v12);
      v17 = v66;
      v15 = (unsigned int)v67;
      v16 = (const __m128i *)&v63;
    }
    else
    {
      v61 = (char *)&v63 - v66;
      sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 0x18u, v18, v12);
      v17 = v66;
      v15 = (unsigned int)v67;
      v16 = (const __m128i *)&v61[v66];
    }
  }
  v19 = (__m128i *)(v17 + 24 * v15);
  *v19 = _mm_loadu_si128(v16);
  v20 = a1 + 120;
  v19[1].m128i_i64[0] = v16[1].m128i_i64[0];
  v21 = v67;
  LODWORD(v67) = v67 + 1;
  if ( v21 >= 0 )
  {
    v22 = 24LL * v21;
    v23 = 24 * (v21 - (unsigned __int64)(unsigned int)v21);
    do
    {
      v24 = sub_317E540(v20, (_DWORD *)(v22 + v66), *(int **)(v22 + v66 + 8), *(_QWORD *)(v22 + v66 + 16));
      v20 = (__int64)v24;
      if ( v23 == v22 )
        break;
      v22 -= 24;
    }
    while ( v24 );
  }
  if ( (_BYTE *)v66 != v68 )
    _libc_free(v66);
  return v20;
}

// Function: sub_C1C070
// Address: 0xc1c070
//
__int64 __fastcall sub_C1C070(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r14
  unsigned __int8 v6; // dl
  bool v7; // al
  __int64 v9; // rcx
  __int64 v10; // rbx
  unsigned __int8 **v11; // rdx
  unsigned __int8 *v12; // rax
  unsigned __int8 v13; // dl
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  int v21; // eax
  __int64 *v22; // rcx
  __int64 v23; // rsi
  _BYTE *v24; // rdi
  int v25; // ebx
  __int64 v26; // r13
  _QWORD *v27; // rax
  unsigned __int8 v28; // al
  unsigned __int8 **v29; // r13
  unsigned __int8 *v30; // rax
  unsigned __int8 v31; // dl
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rax
  const __m128i *v36; // r13
  __m128i *v37; // rax
  __int64 v38; // rdx
  char *v39; // r13
  __int64 v42; // [rsp+28h] [rbp-158h]
  _QWORD v43[4]; // [rsp+30h] [rbp-150h] BYREF
  unsigned __int64 v44; // [rsp+50h] [rbp-130h] BYREF
  __int64 v45; // [rsp+58h] [rbp-128h]
  _BYTE v46[288]; // [rsp+60h] [rbp-120h] BYREF

  v4 = a2;
  v6 = *(_BYTE *)(a2 - 16);
  v44 = (unsigned __int64)v46;
  v45 = 0xA00000000LL;
  v7 = (v6 & 2) != 0;
  if ( (v6 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) != 2 )
      return a1;
    v9 = *(_QWORD *)(a2 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) != 2 )
      return a1;
    v9 = a2 - 16 - 8LL * ((v6 >> 2) & 0xF);
  }
  v10 = *(_QWORD *)(v9 + 8);
  if ( !v10 )
    return a1;
  while ( 1 )
  {
    v42 = v4 - 16;
    v11 = v7 ? *(unsigned __int8 ***)(v4 - 32) : (unsigned __int8 **)(v42 - 8LL * ((*(_BYTE *)(v4 - 16) >> 2) & 0xF));
    v12 = sub_AF34D0(*v11);
    v13 = *(v12 - 16);
    if ( (v13 & 2) != 0 )
    {
      v14 = *(_QWORD *)(*((_QWORD *)v12 - 4) + 24LL);
      if ( !v14 )
        goto LABEL_27;
    }
    else
    {
      v14 = *(_QWORD *)&v12[-8 * ((v13 >> 2) & 0xF) + 8];
      if ( !v14 )
        goto LABEL_27;
    }
    v15 = sub_B91420(v14);
    v17 = v16;
    if ( v16 )
      goto LABEL_11;
LABEL_27:
    v28 = *(_BYTE *)(v4 - 16);
    v29 = (v28 & 2) != 0 ? *(unsigned __int8 ***)(v4 - 32) : (unsigned __int8 **)(v42 - 8LL * ((v28 >> 2) & 0xF));
    v30 = sub_AF34D0(*v29);
    v31 = *(v30 - 16);
    if ( (v31 & 2) != 0 )
    {
      v15 = *(_QWORD *)(*((_QWORD *)v30 - 4) + 16LL);
      if ( v15 )
        goto LABEL_31;
    }
    else
    {
      v15 = *(_QWORD *)&v30[-8 * ((v31 >> 2) & 0xF)];
      if ( v15 )
      {
LABEL_31:
        v15 = sub_B91420(v15);
        v17 = v32;
        goto LABEL_11;
      }
    }
    v17 = 0;
LABEL_11:
    v18 = sub_C1B090(v10, unk_4F838D0);
    v19 = (unsigned int)v45;
    v20 = v18;
    v21 = v45;
    if ( (unsigned int)v45 >= (unsigned __int64)HIDWORD(v45) )
    {
      v34 = (unsigned int)v45 + 1LL;
      v43[1] = v15;
      v35 = v44;
      v43[0] = v20;
      v36 = (const __m128i *)v43;
      v43[2] = v17;
      if ( HIDWORD(v45) < v34 )
      {
        if ( v44 > (unsigned __int64)v43 || (unsigned __int64)v43 >= v44 + 24LL * (unsigned int)v45 )
        {
          sub_C8D5F0(&v44, v46, v34, 24);
          v35 = v44;
          v19 = (unsigned int)v45;
          v36 = (const __m128i *)v43;
        }
        else
        {
          v39 = (char *)v43 - v44;
          sub_C8D5F0(&v44, v46, v34, 24);
          v35 = v44;
          v19 = (unsigned int)v45;
          v36 = (const __m128i *)&v39[v44];
        }
      }
      v37 = (__m128i *)(v35 + 24 * v19);
      *v37 = _mm_loadu_si128(v36);
      v38 = v36[1].m128i_i64[0];
      LODWORD(v45) = v45 + 1;
      v37[1].m128i_i64[0] = v38;
    }
    else
    {
      v22 = (__int64 *)(v44 + 24LL * (unsigned int)v45);
      if ( v22 )
      {
        v22[2] = v17;
        *v22 = v20;
        v22[1] = v15;
        v21 = v45;
      }
      LODWORD(v45) = v21 + 1;
    }
    v7 = (*(_BYTE *)(v10 - 16) & 2) != 0;
    if ( (*(_BYTE *)(v10 - 16) & 2) != 0 )
    {
      if ( *(_DWORD *)(v10 - 24) != 2 )
        break;
      v33 = *(_QWORD *)(v10 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v10 - 16) >> 6) & 0xF) != 2 )
        break;
      v33 = v10 - 16 - 8LL * ((*(_BYTE *)(v10 - 16) >> 2) & 0xF);
    }
    v4 = v10;
    if ( !*(_QWORD *)(v33 + 8) )
      break;
    v10 = *(_QWORD *)(v33 + 8);
  }
  v23 = v44;
  v24 = (_BYTE *)v44;
  if ( (_DWORD)v45 )
  {
    v25 = v45 - 1;
    if ( (int)v45 - 1 >= 0 )
    {
      v26 = 24LL * v25;
      while ( 1 )
      {
        v23 += v26;
        v26 -= 24;
        v27 = sub_C1BBC0(a1, v23, *(_QWORD *)(v23 + 8), *(_QWORD *)(v23 + 16), a3, a4);
        a1 = (__int64)v27;
        if ( --v25 < 0 || !v27 )
          break;
        v23 = v44;
      }
      v24 = (_BYTE *)v44;
    }
  }
  if ( v24 != v46 )
    _libc_free(v24, v23);
  return a1;
}

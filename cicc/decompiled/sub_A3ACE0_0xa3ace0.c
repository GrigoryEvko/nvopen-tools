// Function: sub_A3ACE0
// Address: 0xa3ace0
//
__int64 __fastcall sub_A3ACE0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, char a5, __m128i *a6)
{
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // ecx
  int v12; // edx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _DWORD *v16; // r12
  _QWORD *v17; // rdi
  _DWORD *v18; // rcx
  unsigned __int64 v19; // r8
  char *v20; // rcx
  char *v21; // r9
  unsigned int v22; // edx
  unsigned int v23; // ecx
  __int64 v24; // rsi
  unsigned int v25; // edx
  char *v26; // rax
  _BYTE *v27; // rdx
  int v28; // eax
  unsigned __int64 i; // rdx
  unsigned __int64 v30; // rdx
  void *v31; // rsi
  __int64 result; // rax
  unsigned __int64 v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // r9
  size_t v36; // r8
  char *v37; // r9
  unsigned __int64 v38; // rcx
  unsigned int v39; // edx
  unsigned int v40; // edx
  unsigned int v41; // eax
  __int64 v42; // rsi
  unsigned __int64 v43; // [rsp+0h] [rbp-180h]
  size_t n; // [rsp+8h] [rbp-178h]
  char *na; // [rsp+8h] [rbp-178h]
  unsigned __int8 v49; // [rsp+2Ch] [rbp-154h]
  void *src; // [rsp+30h] [rbp-150h] BYREF
  unsigned __int64 v51; // [rsp+38h] [rbp-148h]
  unsigned __int64 v52; // [rsp+40h] [rbp-140h]
  _BYTE v53[8]; // [rsp+48h] [rbp-138h] BYREF
  _QWORD *v54; // [rsp+50h] [rbp-130h] BYREF
  _QWORD v55[2]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v56; // [rsp+70h] [rbp-110h]
  __int64 v57; // [rsp+78h] [rbp-108h]
  __int64 v58; // [rsp+80h] [rbp-100h]
  __int64 v59[30]; // [rsp+90h] [rbp-F0h] BYREF

  v9 = *(_BYTE **)(a1 + 232);
  v10 = (__int64)&v9[*(_QWORD *)(a1 + 240)];
  v54 = v55;
  sub_A15D40((__int64 *)&v54, v9, v10);
  v11 = *(_DWORD *)(a1 + 276);
  v12 = *(_DWORD *)(a1 + 284);
  v56 = *(_QWORD *)(a1 + 264);
  v57 = *(_QWORD *)(a1 + 272);
  v58 = *(_QWORD *)(a1 + 280);
  LOBYTE(v13) = 0;
  if ( v11 <= 0x1F )
    v13 = (0xD8000222uLL >> v11) & 1;
  v49 = a3;
  if ( v12 != 5 && !(_BYTE)v13 )
  {
    sub_A182C0((__int64)v59, a2);
    sub_A38520(v59, a1, a3, a4, a5, a6);
    sub_A1BA60((__int64)v59);
    sub_A1BEE0((__int64)v59);
    result = sub_A18460((__int64)v59, a1);
    goto LABEL_34;
  }
  v51 = 0;
  src = v53;
  v52 = 0;
  sub_C8D290(&src, v53, 0x40000, 1);
  v14 = v51;
  if ( !v51 )
  {
    if ( v52 <= 0x13 )
    {
      sub_C8D290(&src, v53, 20, 1);
      v14 = v51;
    }
    v26 = (char *)src + v14;
    *(_OWORD *)v26 = 0;
    v51 += 20LL;
    *((_DWORD *)v26 + 4) = 0;
    goto LABEL_22;
  }
  v15 = v51 + 20;
  if ( v51 + 20 > v52 )
  {
    sub_C8D290(&src, v53, v15, 1);
    v14 = v51;
    v16 = src;
    v15 = v51 + 20;
    v17 = (char *)src + v51;
    if ( v51 <= 0x13 )
      goto LABEL_8;
    v36 = v51 - 20;
    v37 = (char *)src + v51 - 20;
    if ( v15 > v52 )
    {
      v43 = v51 - 20;
      na = (char *)src + v51 - 20;
      sub_C8D290(&src, v53, v15, 1);
      v36 = v43;
      v37 = na;
      v17 = (char *)src + v51;
    }
LABEL_53:
    n = v36;
    memmove(v17, v37, 0x14u);
    v51 += 20LL;
    if ( n )
      memmove(v16 + 5, v16, n);
    v16[4] = 0;
    *(_OWORD *)v16 = 0;
    goto LABEL_22;
  }
  v16 = src;
  v17 = (char *)src + v51;
  if ( v51 > 0x13 )
  {
    v36 = v51 - 20;
    v37 = (char *)src + v51 - 20;
    goto LABEL_53;
  }
LABEL_8:
  v51 = v15;
  if ( v17 != (_QWORD *)v16 )
  {
    v18 = v16 + 5;
    if ( (unsigned int)v14 < 8 )
    {
      if ( (v14 & 4) != 0 )
      {
        *v18 = *v16;
        *(_DWORD *)((char *)v18 + (unsigned int)v14 - 4) = *(_DWORD *)((char *)v16 + (unsigned int)v14 - 4);
      }
      else if ( (_DWORD)v14 )
      {
        *(_BYTE *)v18 = *(_BYTE *)v16;
        if ( (v14 & 2) != 0 )
          *(_WORD *)((char *)v18 + (unsigned int)v14 - 2) = *(_WORD *)((char *)v16 + (unsigned int)v14 - 2);
      }
    }
    else
    {
      v19 = (unsigned __int64)(v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v16 + 5) = *(_QWORD *)v16;
      *(_QWORD *)((char *)v18 + (unsigned int)v14 - 8) = *(_QWORD *)((char *)v16 + (unsigned int)v14 - 8);
      v20 = (char *)v18 - v19;
      v21 = (char *)((char *)v16 - v20);
      if ( (((_DWORD)v14 + (_DWORD)v20) & 0xFFFFFFF8) >= 8 )
      {
        v22 = (v14 + (_DWORD)v20) & 0xFFFFFFF8;
        v23 = 0;
        do
        {
          v24 = v23;
          v23 += 8;
          *(_QWORD *)(v19 + v24) = *(_QWORD *)&v21[v24];
        }
        while ( v23 < v22 );
      }
    }
  }
  if ( v14 )
  {
    if ( (unsigned int)v14 >= 8 )
    {
      *(_QWORD *)v16 = 0;
      *(_QWORD *)((char *)v16 + (unsigned int)v14 - 8) = 0;
      v33 = (unsigned __int64)(v16 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (((_DWORD)v14 + (_DWORD)v16 - (_DWORD)v33) & 0xFFFFFFF8) >= 8 )
      {
        v34 = 0;
        do
        {
          v35 = v34;
          v34 += 8;
          *(_QWORD *)(v33 + v35) = 0;
        }
        while ( v34 < (((_DWORD)v14 + (_DWORD)v16 - (_DWORD)v33) & 0xFFFFFFF8) );
      }
    }
    else if ( (v14 & 4) != 0 )
    {
      *v16 = 0;
      *(_DWORD *)((char *)v16 + (unsigned int)v14 - 4) = 0;
    }
    else if ( (_DWORD)v14 )
    {
      *(_BYTE *)v16 = 0;
      if ( (v14 & 2) != 0 )
        *(_WORD *)((char *)v16 + (unsigned int)v14 - 2) = 0;
    }
  }
  v25 = 20 - v14;
  if ( (unsigned int)(20 - v14) >= 8 )
  {
    *v17 = 0;
    *(_QWORD *)((char *)v17 + v25 - 8) = 0;
    v38 = (unsigned __int64)(v17 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    v39 = ((_DWORD)v17 - v38 + v25) & 0xFFFFFFF8;
    if ( v39 >= 8 )
    {
      v40 = v39 & 0xFFFFFFF8;
      v41 = 0;
      do
      {
        v42 = v41;
        v41 += 8;
        *(_QWORD *)(v38 + v42) = 0;
      }
      while ( v41 < v40 );
    }
  }
  else if ( (v25 & 4) != 0 )
  {
    *(_DWORD *)v17 = 0;
    *(_DWORD *)((char *)v17 + v25 - 4) = 0;
  }
  else if ( v25 )
  {
    *(_BYTE *)v17 = 0;
    if ( (v25 & 2) != 0 )
      *(_WORD *)((char *)v17 + v25 - 2) = 0;
  }
LABEL_22:
  sub_A18170((__int64)v59, (__int64)&src);
  sub_A38520(v59, a1, v49, a4, a5, a6);
  sub_A1BA60((__int64)v59);
  sub_A1BEE0((__int64)v59);
  v27 = &algn_1000005[2];
  if ( (_DWORD)v56 != 39 )
  {
    LODWORD(v27) = 7;
    if ( (_DWORD)v56 != 38 )
    {
      LODWORD(v27) = 18;
      if ( (_DWORD)v56 != 22 )
      {
        LODWORD(v27) = 16777234;
        if ( (_DWORD)v56 != 24 )
        {
          if ( (_DWORD)v56 == 1 || (LODWORD(v27) = -1, (_DWORD)v56 == 36) )
            LODWORD(v27) = 12;
        }
      }
    }
  }
  v28 = v51;
  *(_DWORD *)src = 186106078;
  *((_DWORD *)src + 1) = 0;
  *((_DWORD *)src + 2) = 20;
  *((_DWORD *)src + 3) = v28 - 20;
  *((_DWORD *)src + 4) = (_DWORD)v27;
  for ( i = v51; (i & 0xF) != 0; v51 = i )
  {
    v30 = i + 1;
    if ( v30 > v52 )
      sub_C8D290(&src, v53, v30, 1);
    *((_BYTE *)src + v51) = 0;
    i = v51 + 1;
  }
  v31 = src;
  sub_CB6200(a2, src, i);
  result = sub_A18460((__int64)v59, (__int64)v31);
  if ( src != v53 )
    result = _libc_free(src, v31);
LABEL_34:
  if ( v54 != v55 )
    return j_j___libc_free_0(v54, v55[0] + 1LL);
  return result;
}

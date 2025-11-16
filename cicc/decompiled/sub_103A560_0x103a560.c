// Function: sub_103A560
// Address: 0x103a560
//
void __fastcall sub_103A560(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  unsigned __int8 v6; // dl
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 i; // r14
  char *v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  unsigned __int8 v13; // di
  bool v14; // cl
  unsigned int v15; // eax
  unsigned int j; // ebx
  __int64 v17; // rax
  _BYTE *v18; // rax
  unsigned __int8 v19; // dl
  _QWORD *v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __m128i *v26; // rsi
  __int64 v27; // rbx
  char *v28; // rax
  char *v29; // r15
  __int64 v30; // rcx
  __m128i *v31; // rdx
  __m128i *v32; // rax
  _QWORD *v33; // r15
  signed __int64 v34; // r14
  char v35; // al
  void *src; // [rsp+10h] [rbp-90h] BYREF
  char *v37; // [rsp+18h] [rbp-88h]
  char *v38; // [rsp+20h] [rbp-80h]
  __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  __m128i *v40; // [rsp+38h] [rbp-68h]
  __m128i *v41; // [rsp+40h] [rbp-60h]
  __m128i v42; // [rsp+50h] [rbp-50h] BYREF
  __m128i *v43; // [rsp+60h] [rbp-40h]

  v4 = sub_10390E0(a2);
  v38 = 0;
  src = 0;
  v5 = v4;
  v37 = 0;
  v6 = *(_BYTE *)(v4 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(unsigned int *)(v4 - 24);
    if ( !v7 )
    {
LABEL_3:
      v8 = *(_QWORD *)(v5 - 32);
      goto LABEL_4;
    }
  }
  else
  {
    v7 = (*(_WORD *)(v4 - 16) >> 6) & 0xF;
    if ( !(_DWORD)v7 )
      goto LABEL_45;
  }
  v27 = 8 * v7;
  v28 = (char *)sub_22077B0(8 * v7);
  v29 = v28;
  if ( v37 - (_BYTE *)src > 0 )
  {
    memmove(v28, src, v37 - (_BYTE *)src);
    j_j___libc_free_0(src, v38 - (_BYTE *)src);
  }
  src = v29;
  v37 = v29;
  v38 = &v29[v27];
  v6 = *(_BYTE *)(v5 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(unsigned int *)(v5 - 24);
    goto LABEL_3;
  }
  v7 = (*(_WORD *)(v5 - 16) >> 6) & 0xF;
LABEL_45:
  v8 = v5 - 8LL * ((v6 >> 2) & 0xF) - 16;
LABEL_4:
  for ( i = v8 + 8 * v7; i != v8; v37 = v10 + 8 )
  {
    while ( 1 )
    {
      if ( **(_BYTE **)v8 != 1 || (v11 = *(_QWORD *)(*(_QWORD *)v8 + 136LL), *(_BYTE *)v11 != 17) )
        BUG();
      v12 = *(_QWORD **)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) > 0x40u )
        v12 = (_QWORD *)*v12;
      v42.m128i_i64[0] = (__int64)v12;
      v10 = v37;
      if ( v37 != v38 )
        break;
      v8 += 8;
      sub_A235E0((__int64)&src, v37, &v42);
      if ( i == v8 )
        goto LABEL_15;
    }
    if ( v37 )
    {
      *(_QWORD *)v37 = v12;
      v10 = v37;
    }
    v8 += 8;
  }
LABEL_15:
  v13 = *(_BYTE *)(a2 - 16);
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v14 = (v13 & 2) != 0;
  if ( (v13 & 2) != 0 )
    v15 = *(_DWORD *)(a2 - 24);
  else
    v15 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( v15 > 2 )
  {
    for ( j = 2; ; ++j )
    {
      if ( v14 )
      {
        if ( j >= *(_DWORD *)(a2 - 24) )
          goto LABEL_47;
        v17 = *(_QWORD *)(a2 - 32);
      }
      else
      {
        if ( j >= ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) )
        {
LABEL_47:
          v30 = v39;
          v31 = v40;
          v32 = v41;
          goto LABEL_48;
        }
        v17 = a2 + -16 - 8LL * ((v13 >> 2) & 0xF);
      }
      v18 = *(_BYTE **)(v17 + 8LL * j);
      if ( (unsigned __int8)(*v18 - 5) > 0x1Fu )
        BUG();
      v19 = *(v18 - 16);
      if ( (v19 & 2) != 0 )
        v20 = (_QWORD *)*((_QWORD *)v18 - 4);
      else
        v20 = &v18[-16 - 8LL * ((v19 >> 2) & 0xF)];
      if ( *(_BYTE *)*v20 != 1 || (v21 = *(_QWORD *)(*v20 + 136LL), *(_BYTE *)v21 != 17) )
        BUG();
      v22 = *(_QWORD **)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        v22 = (_QWORD *)*v22;
      v23 = v20[1];
      if ( *(_BYTE *)v23 != 1 || (v24 = *(_QWORD *)(v23 + 136), *(_BYTE *)v24 != 17) )
        BUG();
      v25 = *(_QWORD **)(v24 + 24);
      if ( *(_DWORD *)(v24 + 32) > 0x40u )
        v25 = (_QWORD *)*v25;
      v42.m128i_i64[0] = (__int64)v22;
      v26 = v40;
      v42.m128i_i64[1] = (__int64)v25;
      if ( v40 == v41 )
      {
        sub_9D3F90((__int64)&v39, v40, &v42);
        v13 = *(_BYTE *)(a2 - 16);
        v14 = (v13 & 2) != 0;
      }
      else
      {
        if ( v40 )
        {
          *v40 = _mm_loadu_si128(&v42);
          v13 = *(_BYTE *)(a2 - 16);
          v26 = v40;
          v14 = (v13 & 2) != 0;
        }
        v40 = v26 + 1;
      }
    }
  }
  v32 = 0;
  v31 = 0;
  v30 = 0;
LABEL_48:
  v33 = src;
  v42.m128i_i64[0] = v30;
  v42.m128i_i64[1] = (__int64)v31;
  v43 = v32;
  v41 = 0;
  v34 = (v37 - (_BYTE *)src) >> 3;
  v40 = 0;
  v39 = 0;
  v35 = sub_1039110(a2);
  sub_103A310(a1, v35, v33, v34, (char **)&v42);
  if ( v42.m128i_i64[0] )
    j_j___libc_free_0(v42.m128i_i64[0], (char *)v43 - v42.m128i_i64[0]);
  if ( v39 )
    j_j___libc_free_0(v39, (char *)v41 - v39);
  if ( src )
    j_j___libc_free_0(src, v38 - (_BYTE *)src);
}

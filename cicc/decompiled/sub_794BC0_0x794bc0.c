// Function: sub_794BC0
// Address: 0x794bc0
//
__int64 __fastcall sub_794BC0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  _QWORD *v6; // rbx
  bool v7; // zf
  int v8; // ecx
  unsigned int v9; // edx
  unsigned int v10; // eax
  __int64 v11; // r14
  size_t v12; // rdx
  __int64 v13; // rax
  char *v14; // rcx
  __int16 *v15; // r14
  unsigned int v17; // ecx
  __int64 v18; // rsi
  unsigned int v19; // edx
  __m128i *v20; // rax
  __m128i v21; // xmm0
  __m128i *v22; // rax
  __int64 v23; // rax
  size_t v24; // [rsp+0h] [rbp-110h]
  size_t v25; // [rsp+0h] [rbp-110h]
  int v26; // [rsp+Ch] [rbp-104h]
  unsigned int v27; // [rsp+Ch] [rbp-104h]
  unsigned int i; // [rsp+1Ch] [rbp-F4h] BYREF
  __int64 v29; // [rsp+20h] [rbp-F0h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-E8h]
  int v31; // [rsp+2Ch] [rbp-E4h]
  void *s; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-D8h]
  __int64 v34; // [rsp+40h] [rbp-D0h]
  int v35; // [rsp+48h] [rbp-C8h]
  char v36; // [rsp+78h] [rbp-98h] BYREF
  _QWORD v37[18]; // [rsp+80h] [rbp-90h] BYREF

  v6 = *(_QWORD **)a1;
  for ( i = 1; *((_BYTE *)v6 + 140) == 12; v6 = (_QWORD *)v6[20] )
    ;
  if ( dword_4F08058 )
  {
    sub_771BE0();
    dword_4F08058 = 0;
  }
  sub_774A30((__int64)&v29, 1);
  if ( !(unsigned int)sub_8D29A0(v6) || (*(_BYTE *)(a1 + 25) & 3) != 0 )
    goto LABEL_26;
  v7 = *(_QWORD *)(a2 + 16) == 0;
  v37[2] = *(_QWORD *)(a1 + 28);
  if ( v7 )
    goto LABEL_8;
  v17 = v30;
  v18 = v29;
  v19 = v30 & ((unsigned __int64)&v36 >> 3);
  v20 = (__m128i *)(v29 + 16LL * v19);
  if ( v20->m128i_i64[0] )
  {
    v21 = _mm_loadu_si128(v20);
    v20->m128i_i64[0] = (__int64)&v36;
    v20->m128i_i64[1] = a2;
    do
    {
      v19 = v17 & (v19 + 1);
      v22 = (__m128i *)(v18 + 16LL * v19);
    }
    while ( v22->m128i_i64[0] );
    *v22 = v21;
  }
  else
  {
    v20->m128i_i64[0] = (__int64)&v36;
    v20->m128i_i64[1] = a2;
  }
  ++v31;
  if ( 2 * v31 > v17 )
    sub_7704A0((__int64)&v29);
  if ( (*(_BYTE *)(a1 + 25) & 3) == 0 )
  {
LABEL_8:
    v8 = 16;
    if ( (unsigned __int8)(*((_BYTE *)v6 + 140) - 2) > 1u )
      v8 = sub_7764B0((__int64)&v29, (unsigned __int64)v6, &i);
  }
  else
  {
    v8 = 32;
  }
  if ( i )
  {
    if ( (unsigned __int8)(*((_BYTE *)v6 + 140) - 8) > 3u )
    {
      v12 = 8;
      v11 = 16;
      v10 = 16;
    }
    else
    {
      v9 = (unsigned int)(v8 + 7) >> 3;
      v10 = v9 + 9;
      if ( (((_BYTE)v9 + 9) & 7) != 0 )
        v10 = v9 + 17 - (((_BYTE)v9 + 9) & 7);
      v11 = v10;
      v12 = v10 - 8LL;
    }
    v13 = v8 + v10;
    if ( (unsigned int)v13 > 0x400 )
    {
      v24 = v12;
      v26 = v13 + 16;
      v23 = sub_822B10((unsigned int)(v13 + 16));
      v12 = v24;
      *(_QWORD *)v23 = v34;
      *(_DWORD *)(v23 + 8) = v26;
      *(_DWORD *)(v23 + 12) = v35;
      v14 = (char *)(v23 + 16);
      v34 = v23;
    }
    else
    {
      if ( (v13 & 7) != 0 )
        v13 = (_DWORD)v13 + 8 - (unsigned int)(v13 & 7);
      v14 = (char *)s;
      if ( 0x10000 - ((int)s - (int)v33) < (unsigned int)v13 )
      {
        v25 = v12;
        v27 = v13;
        sub_772E70(&s);
        v14 = (char *)s;
        v12 = v25;
        v13 = v27;
      }
      s = &v14[v13];
    }
    v15 = (__int16 *)((char *)memset(v14, 0, v12) + v11);
    *((_QWORD *)v15 - 1) = v6;
    if ( (unsigned __int8)(*((_BYTE *)v6 + 140) - 9) <= 2u )
      *(_QWORD *)v15 = 0;
    if ( (unsigned int)sub_786210((__int64)&v29, (_QWORD **)a1, (unsigned __int64)v15, (char *)v15) )
    {
      if ( !(unsigned int)sub_621000(v15, 0, (__int16 *)&xmmword_4F08290, 0) )
        *a4 = 0;
      goto LABEL_27;
    }
LABEL_26:
    i = 0;
  }
LABEL_27:
  sub_67E3D0(v37);
  sub_771990((__int64)&v29);
  return i;
}

// Function: sub_9365F0
// Address: 0x9365f0
//
__m128i *__fastcall sub_9365F0(
        __m128i *a1,
        __int64 a2,
        unsigned int *a3,
        int a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int8 a7)
{
  __int64 v7; // r15
  __m128i *v8; // r12
  int v9; // ebx
  _QWORD *v10; // r14
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  char v15; // al
  __m128i v16; // xmm0
  bool v17; // zf
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // r12
  int v22; // r15d
  __int64 v23; // rbx
  __int64 v24; // rsi
  _QWORD *v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  _BYTE *v28; // rsi
  _QWORD *i; // r13
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // [rsp+8h] [rbp-88h]
  char v36; // [rsp+10h] [rbp-80h]
  __m128i *v37; // [rsp+10h] [rbp-80h]
  unsigned __int64 j; // [rsp+38h] [rbp-58h] BYREF
  __m128i v42; // [rsp+40h] [rbp-50h] BYREF
  __int64 v43; // [rsp+50h] [rbp-40h]

  v7 = a2;
  v8 = a1;
  v9 = a4;
  v36 = a4;
  if ( *(_BYTE *)(a2 + 240) )
  {
    v42 = 0u;
    v31 = *(_QWORD *)(a2 + 424);
    v43 = 0;
    if ( v31 == *(_QWORD *)(v7 + 432) )
    {
      sub_932050((char **)(v7 + 416), (char *)v31, v42.m128i_i64);
      if ( v42.m128i_i64[0] )
        j_j___libc_free_0(v42.m128i_i64[0], v43 - v42.m128i_i64[0]);
    }
    else
    {
      if ( v31 )
      {
        *(_QWORD *)v31 = 0;
        *(_QWORD *)(v31 + 8) = v42.m128i_i64[1];
        *(_QWORD *)(v31 + 16) = v43;
        v31 = *(_QWORD *)(v7 + 424);
      }
      *(_QWORD *)(v7 + 424) = v31 + 24;
    }
  }
  v10 = *(_QWORD **)(v7 + 528);
  v35 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 368LL);
  if ( (unsigned int *)v10[10] != a3 )
  {
    v10 = *(_QWORD **)(*((_QWORD *)a3 + 10) + 8LL);
    if ( dword_4D04658 )
    {
      if ( !v10 )
        goto LABEL_18;
      goto LABEL_5;
    }
LABEL_63:
    sub_941230(v35, *a3);
    sub_9415C0(v35, v7 + 48);
    if ( !v10 )
      goto LABEL_18;
    goto LABEL_5;
  }
  if ( !dword_4D04658 )
    goto LABEL_63;
LABEL_5:
  if ( dword_4D046B4 )
  {
    for ( i = (_QWORD *)v10[16]; i; i = (_QWORD *)i[2] )
    {
      v30 = sub_916620(*(__int64 **)(v7 + 32), i[1], 0);
      if ( v30 )
        sub_943430(*(_QWORD *)(*(_QWORD *)(v7 + 32) + 368LL), v30, i[1], *i);
    }
  }
  v11 = v10[14];
  for ( j = v11; v11; j = v11 )
  {
    if ( (*(_BYTE *)(v11 + 170) & 0x60) == 0
      && *(_BYTE *)(v11 + 177) != 5
      && *(_BYTE *)(*(_QWORD *)(v11 + 120) + 140LL) != 14 )
    {
      sub_943C40(v7);
    }
    v11 = *(_QWORD *)(j + 112);
  }
  v12 = v10[15];
  for ( j = v12; v12; j = v12 )
  {
    if ( (*(_BYTE *)(v12 + 170) & 0x60) == 0 && *(_BYTE *)(v12 + 177) != 5 )
      sub_944CC0(v7);
    v12 = *(_QWORD *)(j + 112);
  }
  if ( *(_BYTE *)(v7 + 240) )
  {
    v26 = v10[15];
    for ( j = v26; v26; j = v26 )
    {
      if ( (*(_BYTE *)(v26 + 170) & 0x60) == 0 && *(_BYTE *)(v26 + 177) != 5 )
      {
        if ( !sub_91CC40(v7 + 248, v26) )
        {
          v27 = *(_QWORD *)(v7 + 424);
          v28 = *(_BYTE **)(v27 - 16);
          if ( v28 == *(_BYTE **)(v27 - 8) )
          {
            sub_930F50(v27 - 24, v28, &j);
          }
          else
          {
            if ( v28 )
            {
              *(_QWORD *)v28 = j;
              v28 = *(_BYTE **)(v27 - 16);
            }
            *(_QWORD *)(v27 - 16) = v28 + 8;
          }
        }
        if ( !sub_91CC40(v7 + 248, j) || sub_91CCF0(v7 + 248, j) )
          sub_9465D0(v7, j, 0);
      }
      v26 = *(_QWORD *)(j + 112);
    }
  }
LABEL_18:
  if ( !v36 )
  {
    a1->m128i_i8[12] &= ~1u;
    a1->m128i_i64[0] = 0;
    a1->m128i_i32[2] = 0;
    a1[1].m128i_i32[0] = 0;
  }
  v13 = *((_QWORD *)a3 + 9);
  if ( v13 )
  {
    while ( 1 )
    {
      if ( (_BYTE)v9 && !*(_QWORD *)(v13 + 16) )
        goto LABEL_27;
LABEL_23:
      sub_9363D0((_QWORD *)v7, v13);
      if ( !*(_BYTE *)(v7 + 240) )
        goto LABEL_33;
      while ( 1 )
      {
        v14 = *(_QWORD *)(v13 + 16);
        if ( !v14 )
          break;
        v13 = *(_QWORD *)(v13 + 16);
        if ( !(_BYTE)v9 || *(_QWORD *)(v14 + 16) )
          goto LABEL_23;
LABEL_27:
        v15 = *(_BYTE *)(v13 + 40);
        if ( v15 && v15 != 25 )
          sub_91B8A0("unexpected: last statement in statement expression is notan expression!", (_DWORD *)v13, 1);
        if ( !*(_QWORD *)(v7 + 96) )
        {
          v34 = (_QWORD *)sub_945CA0(v7, byte_3F871B3, 0, 0);
          sub_92FEA0(v7, v34, 0);
        }
        sub_921EA0((__int64)&v42, v7, *(__int64 **)(v13 + 48), a5, a6, a7);
        v16 = _mm_loadu_si128(&v42);
        v17 = *(_BYTE *)(v7 + 240) == 0;
        v8[1].m128i_i32[0] = v43;
        *v8 = v16;
        if ( v17 )
          goto LABEL_33;
      }
      if ( (unsigned __int8)sub_946900(v7, v13)
        || (v19 = *(_QWORD *)(v7 + 424), v20 = *(_QWORD *)(v19 - 16), v20 == *(_QWORD *)(v19 - 24)) )
      {
LABEL_33:
        v13 = *(_QWORD *)(v13 + 16);
        if ( !v13 )
          break;
      }
      else
      {
        v37 = v8;
        v21 = v7;
        v22 = v9;
        v23 = v19;
        do
        {
          if ( !*(_QWORD *)(v21 + 96) )
          {
            v25 = (_QWORD *)sub_945CA0(v21, byte_3F871B3, 0, 0);
            sub_92FEA0(v21, v25, 0);
          }
          v24 = *(_QWORD *)(v20 - 8);
          v20 -= 8;
          sub_9465D0(v21, v24, 1);
        }
        while ( *(_QWORD *)(v23 - 24) != v20 );
        v13 = *(_QWORD *)(v13 + 16);
        v9 = v22;
        v7 = v21;
        v8 = v37;
        if ( !v13 )
          break;
      }
    }
  }
  if ( !dword_4D04658 )
    sub_93FF00(v35, v7 + 48);
  if ( *(_BYTE *)(v7 + 240) )
  {
    v32 = *(_QWORD *)(v7 + 424);
    *(_QWORD *)(v7 + 424) = v32 - 24;
    v33 = *(_QWORD *)(v32 - 24);
    if ( v33 )
      j_j___libc_free_0(v33, *(_QWORD *)(v32 - 8) - v33);
  }
  return v8;
}

// Function: sub_194A200
// Address: 0x194a200
//
char __fastcall sub_194A200(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v9; // r13
  __m128i *v10; // rax
  char v11; // dl
  __int64 v12; // rdx
  __int64 v13; // r9
  __m128i *v14; // rsi
  unsigned int v15; // edi
  __int64 *v16; // rcx
  __int64 v17; // r14
  int v18; // eax
  __m128i *v19; // r14
  __int64 v20; // r13
  __int64 v21; // rax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  unsigned __int32 v25; // eax
  __int64 v26; // rsi
  unsigned __int32 v27; // edx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  char v31; // al
  unsigned __int32 v32; // r13d
  __int64 v33; // rdx
  char v35; // [rsp+4h] [rbp-8Ch]
  unsigned int v36; // [rsp+8h] [rbp-88h]
  int v37; // [rsp+8h] [rbp-88h]
  _QWORD *v38; // [rsp+8h] [rbp-88h]
  char v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int32 v43; // [rsp+28h] [rbp-68h]
  __m128i v44; // [rsp+30h] [rbp-60h] BYREF
  __m128i v45; // [rsp+40h] [rbp-50h] BYREF
  __int64 v46; // [rsp+50h] [rbp-40h]

  v5 = (__int64)a3;
  v9 = *a3;
  v10 = *(__m128i **)(a5 + 8);
  if ( *(__m128i **)(a5 + 16) == v10 )
    goto LABEL_12;
LABEL_2:
  LOBYTE(v10) = (unsigned __int8)sub_16CCBA0(a5, v9);
  if ( !v11 )
    return (char)v10;
  while ( 1 )
  {
    LOBYTE(v10) = *(_BYTE *)(v9 + 16);
    if ( (_BYTE)v10 == 50 )
      goto LABEL_7;
    if ( (_BYTE)v10 != 5 )
      break;
    if ( *(_WORD *)(v9 + 18) != 26 )
      return (char)v10;
LABEL_7:
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      v12 = *(_QWORD *)(v9 - 8);
    else
      v12 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    sub_194A200(a1, a2, v12, a4, a5);
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      v13 = *(_QWORD *)(v9 - 8);
    else
      v13 = v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    v5 = v13 + 24;
    v10 = *(__m128i **)(a5 + 8);
    v9 = *(_QWORD *)(v13 + 24);
    if ( *(__m128i **)(a5 + 16) != v10 )
      goto LABEL_2;
LABEL_12:
    v14 = (__m128i *)((char *)v10 + 8 * *(unsigned int *)(a5 + 28));
    v15 = *(_DWORD *)(a5 + 28);
    if ( v10 == v14 )
      goto LABEL_24;
    v16 = 0;
    do
    {
      if ( v9 == v10->m128i_i64[0] )
        return (char)v10;
      if ( v10->m128i_i64[0] == -2 )
        v16 = (__int64 *)v10;
      v10 = (__m128i *)((char *)v10 + 8);
    }
    while ( v14 != v10 );
    if ( !v16 )
    {
LABEL_24:
      if ( v15 >= *(_DWORD *)(a5 + 24) )
        goto LABEL_2;
      *(_DWORD *)(a5 + 28) = v15 + 1;
      v14->m128i_i64[0] = v9;
      ++*(_QWORD *)a5;
    }
    else
    {
      *v16 = v9;
      --*(_DWORD *)(a5 + 32);
      ++*(_QWORD *)a5;
    }
  }
  if ( (_BYTE)v10 != 75 )
    return (char)v10;
  v17 = *(_QWORD *)(v9 - 24);
  v40 = *(_QWORD *)(v9 - 48);
  v18 = *(unsigned __int16 *)(v9 + 18);
  BYTE1(v18) &= ~0x80u;
  LODWORD(v10) = v18 - 34;
  switch ( (int)v10 )
  {
    case 0:
      goto LABEL_50;
    case 2:
      v40 = *(_QWORD *)(v9 - 24);
      v17 = *(_QWORD *)(v9 - 48);
LABEL_50:
      v29 = sub_146F1B0(a2, v40);
      LOBYTE(v10) = sub_146CEE0(a2, v29, a1);
      if ( !(_BYTE)v10 )
        return (char)v10;
      v35 = 0;
      v37 = 3;
      goto LABEL_32;
    case 4:
      goto LABEL_41;
    case 5:
      goto LABEL_27;
    case 6:
      v40 = *(_QWORD *)(v9 - 24);
      v17 = *(_QWORD *)(v9 - 48);
LABEL_41:
      if ( *(_BYTE *)(v17 + 16) != 13 )
        goto LABEL_47;
      v25 = *(_DWORD *)(v17 + 32);
      v44.m128i_i32[2] = v25;
      if ( v25 <= 0x40 )
      {
        v26 = *(_QWORD *)(v17 + 24);
LABEL_44:
        v44.m128i_i64[0] = ~v26 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v25);
        goto LABEL_45;
      }
      sub_16A4FD0((__int64)&v44, (const void **)(v17 + 24));
      LOBYTE(v25) = v44.m128i_i8[8];
      if ( v44.m128i_i32[2] <= 0x40u )
      {
        v26 = v44.m128i_i64[0];
        goto LABEL_44;
      }
      sub_16A8F40(v44.m128i_i64);
LABEL_45:
      sub_16A7400((__int64)&v44);
      v27 = v44.m128i_u32[2];
      v44.m128i_i32[2] = 0;
      v43 = v27;
      v42 = v44.m128i_i64[0];
      if ( v27 > 0x40 )
      {
        v38 = (_QWORD *)v44.m128i_i64[0];
        if ( v27 - (unsigned int)sub_16A57B0((__int64)&v42) <= 0x40 && *v38 == 1 )
        {
          j_j___libc_free_0_0(v38);
          if ( v44.m128i_i32[2] <= 0x40u )
            goto LABEL_31;
          v30 = v44.m128i_i64[0];
          v31 = 1;
          if ( !v44.m128i_i64[0] )
            goto LABEL_31;
        }
        else
        {
          if ( !v38 )
            goto LABEL_47;
          j_j___libc_free_0_0(v38);
          if ( v44.m128i_i32[2] <= 0x40u )
            goto LABEL_47;
          v30 = v44.m128i_i64[0];
          if ( !v44.m128i_i64[0] )
            goto LABEL_47;
          v31 = 0;
        }
        v39 = v31;
        j_j___libc_free_0_0(v30);
        if ( !v39 )
          goto LABEL_47;
LABEL_31:
        v17 = v40;
        v35 = 1;
        v40 = 0;
        v37 = 1;
        goto LABEL_32;
      }
      if ( v44.m128i_i64[0] == 1 )
        goto LABEL_31;
LABEL_47:
      v28 = sub_146F1B0(a2, v40);
      LOBYTE(v10) = sub_146CEE0(a2, v28, a1);
      v35 = (char)v10;
      if ( !(_BYTE)v10 )
        return (char)v10;
      v37 = 2;
LABEL_32:
      v10 = (__m128i *)sub_146F1B0(a2, v17);
      v19 = v10;
      if ( v10[1].m128i_i16[4] == 7 )
      {
        LOBYTE(v10) = a1;
        if ( v19[3].m128i_i64[0] == a1 && v19[2].m128i_i64[1] == 2 )
        {
          if ( v40 )
          {
            v20 = sub_146F1B0(a2, v40);
LABEL_37:
            v45.m128i_i64[0] = v20;
            v44.m128i_i64[0] = *(_QWORD *)v19[2].m128i_i64[0];
            v21 = sub_13A5BC0(v19, a2);
            v45.m128i_i64[1] = v5;
            v44.m128i_i64[1] = v21;
            LODWORD(v46) = v37;
            BYTE4(v46) = v35;
            v24 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v24 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 40, v22, v23);
              v24 = *(unsigned int *)(a4 + 8);
            }
            v10 = (__m128i *)(*(_QWORD *)a4 + 40 * v24);
            *v10 = _mm_loadu_si128(&v44);
            v10[1] = _mm_loadu_si128(&v45);
            v10[2].m128i_i64[0] = v46;
            ++*(_DWORD *)(a4 + 8);
            return (char)v10;
          }
          v32 = *(_DWORD *)(sub_1456040(*(_QWORD *)v19[2].m128i_i64[0]) + 8) >> 8;
          v44.m128i_i32[2] = v32;
          if ( v32 > 0x40 )
          {
            sub_16A4EF0((__int64)&v44, -1, 1);
            v33 = ~(1LL << ((unsigned __int8)v32 - 1));
            if ( v44.m128i_i32[2] > 0x40u )
            {
              *(_QWORD *)(v44.m128i_i64[0] + 8LL * ((v32 - 1) >> 6)) &= v33;
              goto LABEL_69;
            }
          }
          else
          {
            v44.m128i_i64[0] = 0xFFFFFFFFFFFFFFFFLL >> -(char)v32;
            v33 = ~(1LL << ((unsigned __int8)v32 - 1));
          }
          v44.m128i_i64[0] &= v33;
LABEL_69:
          v20 = sub_145CF40(a2, (__int64)&v44);
          if ( v44.m128i_i32[2] > 0x40u && v44.m128i_i64[0] )
            j_j___libc_free_0_0(v44.m128i_i64[0]);
          goto LABEL_37;
        }
      }
      return (char)v10;
    case 7:
      v10 = *(__m128i **)(v9 - 48);
      v40 = *(_QWORD *)(v9 - 24);
      v17 = (__int64)v10;
LABEL_27:
      if ( *(_BYTE *)(v17 + 16) != 13 )
        return (char)v10;
      v36 = *(_DWORD *)(v17 + 32);
      if ( v36 > 0x40 )
      {
        LODWORD(v10) = sub_16A57B0(v17 + 24);
        if ( v36 - (unsigned int)v10 > 0x40 )
          return (char)v10;
        v10 = **(__m128i ***)(v17 + 24);
      }
      else
      {
        v10 = *(__m128i **)(v17 + 24);
      }
      if ( v10 )
        return (char)v10;
      goto LABEL_31;
    default:
      return (char)v10;
  }
}

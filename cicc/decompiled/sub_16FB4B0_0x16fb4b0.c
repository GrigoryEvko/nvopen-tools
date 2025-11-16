// Function: sub_16FB4B0
// Address: 0x16fb4b0
//
__int64 __fastcall sub_16FB4B0(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned int v8; // r13d
  char *v9; // r14
  int v10; // ebx
  char *v11; // rsi
  char *v12; // rax
  int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // r11d
  char v19; // al
  char *v20; // rax
  __int64 *v21; // rax
  __m128i v22; // xmm0
  _BYTE *v23; // rsi
  __int64 *v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rbx
  int v28; // r8d
  int v29; // r9d
  _QWORD *v30; // rdi
  char *v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rax
  int v39; // [rsp+4h] [rbp-7Ch]
  char *v40; // [rsp+8h] [rbp-78h]
  unsigned __int8 v41; // [rsp+8h] [rbp-78h]
  unsigned __int8 v42; // [rsp+8h] [rbp-78h]
  const char *v43; // [rsp+10h] [rbp-70h] BYREF
  __m128i v44; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v45; // [rsp+28h] [rbp-58h]
  __int64 v46; // [rsp+30h] [rbp-50h]
  _QWORD v47[9]; // [rsp+38h] [rbp-48h] BYREF

  v5 = a1;
  v6 = *(unsigned int *)(a1 + 60);
  v7 = *(unsigned int *)(a1 + 56);
  v40 = *(char **)(a1 + 40);
  v39 = *(_DWORD *)(a1 + 60);
  if ( *v40 == 35 )
  {
    v9 = *(char **)(a1 + 40);
LABEL_45:
    v43 = "Got empty plain scalar";
    v32 = *(_QWORD *)(v5 + 48);
    v44.m128i_i16[4] = 259;
    if ( v32 <= (unsigned __int64)v9 )
LABEL_46:
      *(_QWORD *)(v5 + 40) = v32 - 1;
LABEL_38:
    v33 = *(_QWORD *)(v5 + 344);
    if ( v33 )
    {
      v34 = sub_2241E50(a1, a2, v7, v6, a5);
      *(_DWORD *)v33 = 22;
      *(_QWORD *)(v33 + 8) = v34;
    }
    if ( !*(_BYTE *)(v5 + 74) )
      sub_16D14E0(*(__int64 **)v5, *(_QWORD *)(v5 + 40), 0, (__int64)&v43, 0, 0, 0, 0, *(_BYTE *)(v5 + 75));
    *(_BYTE *)(v5 + 74) = 1;
    return 0;
  }
  v8 = v7 + 1;
  v9 = *(char **)(a1 + 40);
  v10 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)sub_16F7940(v5, v9) )
        goto LABEL_19;
      v11 = *(char **)(v5 + 40);
      if ( *(_DWORD *)(v5 + 68) )
      {
        if ( *v11 != 58 )
          break;
        v14 = sub_16F7940(v5, v11 + 1);
        v11 = *(char **)(v5 + 40);
        v18 = v14;
        if ( !(_BYTE)v14 && v11[1] != 44 )
        {
          v43 = "Found unexpected ':' while scanning a plain scalar";
          v36 = *(_QWORD *)(v5 + 48);
          v44.m128i_i16[4] = 259;
          if ( (unsigned __int64)v11 >= v36 )
            *(_QWORD *)(v5 + 40) = v36 - 1;
          v37 = *(_QWORD *)(v5 + 344);
          if ( v37 )
          {
            v41 = v18;
            v38 = sub_2241E50(v5, v11, v15, v16, v17);
            *(_DWORD *)v37 = 22;
            v18 = v41;
            *(_QWORD *)(v37 + 8) = v38;
          }
          if ( !*(_BYTE *)(v5 + 74) )
          {
            v42 = v18;
            sub_16D14E0(*(__int64 **)v5, *(_QWORD *)(v5 + 40), 0, (__int64)&v43, 0, 0, 0, 0, *(_BYTE *)(v5 + 75));
            v18 = v42;
          }
          *(_BYTE *)(v5 + 74) = 1;
          return v18;
        }
        if ( *v11 != 58 )
        {
          v13 = *(_DWORD *)(v5 + 68);
          goto LABEL_13;
        }
      }
      else if ( *v11 != 58 )
      {
        goto LABEL_6;
      }
      if ( (unsigned __int8)sub_16F7940(v5, v11 + 1) )
        goto LABEL_19;
      v13 = *(_DWORD *)(v5 + 68);
      v11 = *(char **)(v5 + 40);
LABEL_13:
      if ( v13 )
        break;
LABEL_6:
      v12 = sub_16F6380(v5, v11);
      v9 = v12;
      if ( *(char **)(v5 + 40) == v12 )
        goto LABEL_20;
      ++*(_DWORD *)(v5 + 60);
      *(_QWORD *)(v5 + 40) = v12;
    }
    v43 = v11;
    v44.m128i_i64[0] = 1;
    if ( sub_16D23E0(&v43, ",:?[]{}", 7, 0) == -1 )
    {
      v11 = *(char **)(v5 + 40);
      goto LABEL_6;
    }
LABEL_19:
    v9 = *(char **)(v5 + 40);
LABEL_20:
    a2 = v9;
    a1 = v5;
    v19 = sub_16F7940(v5, v9);
    v9 = *(char **)(v5 + 40);
    if ( !v19 )
      goto LABEL_30;
LABEL_21:
    a2 = v9;
    a1 = v5;
    if ( (unsigned __int8)sub_16F7940(v5, v9) )
    {
      while ( 1 )
      {
        a2 = v9;
        a1 = v5;
        v20 = sub_16F6430(v5, v9);
        if ( v9 == v20 )
        {
          v31 = sub_16F7720(v5, v9);
          *(_DWORD *)(v5 + 60) = 0;
          v9 = v31;
          if ( !v10 )
            v10 = 1;
          ++*(_DWORD *)(v5 + 64);
          goto LABEL_21;
        }
        if ( v10 && v8 > *(_DWORD *)(v5 + 60) && *v9 == 9 )
          break;
        ++*(_DWORD *)(v5 + 60);
        v9 = v20;
        a1 = v5;
        a2 = v20;
        if ( !(unsigned __int8)sub_16F7940(v5, v20) )
          goto LABEL_27;
      }
      v43 = "Found invalid tab character in indentation";
      v32 = *(_QWORD *)(v5 + 48);
      v44.m128i_i16[4] = 259;
      if ( *(_QWORD *)(v5 + 40) >= v32 )
        goto LABEL_46;
      goto LABEL_38;
    }
LABEL_27:
    if ( !*(_DWORD *)(v5 + 68) && *(_DWORD *)(v5 + 60) < v8 )
      break;
    *(_QWORD *)(v5 + 40) = v9;
    if ( *v9 == 35 )
      goto LABEL_30;
  }
  v9 = *(char **)(v5 + 40);
LABEL_30:
  if ( v40 == v9 )
    goto LABEL_45;
  v46 = 0;
  v45 = v47;
  v44.m128i_i64[1] = v9 - v40;
  LOBYTE(v47[0]) = 0;
  LODWORD(v43) = 18;
  v44.m128i_i64[0] = (__int64)v40;
  v21 = (__int64 *)sub_145CBF0((__int64 *)(v5 + 80), 72, 16);
  v22 = _mm_loadu_si128(&v44);
  v23 = v45;
  v24 = v21;
  *v21 = 0;
  v25 = v46;
  v21[1] = 0;
  LODWORD(v21) = (_DWORD)v43;
  *(__m128i *)(v24 + 3) = v22;
  *((_DWORD *)v24 + 4) = (_DWORD)v21;
  v24[5] = (__int64)(v24 + 7);
  sub_16F6740(v24 + 5, v23, (__int64)&v23[v25]);
  v26 = *(_QWORD *)(v5 + 184);
  v24[1] = v5 + 184;
  v26 &= 0xFFFFFFFFFFFFFFF8LL;
  *v24 = v26 | *v24 & 7;
  *(_QWORD *)(v26 + 8) = v24;
  v27 = *(_QWORD *)(v5 + 184) & 7LL | (unsigned __int64)v24;
  *(_QWORD *)(v5 + 184) = v27;
  sub_16F79B0(v5, v27 & 0xFFFFFFFFFFFFFFF8LL, v39, 0, v28, v29);
  v30 = v45;
  *(_BYTE *)(v5 + 73) = 0;
  if ( v30 != v47 )
    j_j___libc_free_0(v30, v47[0] + 1LL);
  return 1;
}

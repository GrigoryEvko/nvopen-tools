// Function: sub_2BE7B90
// Address: 0x2be7b90
//
__int64 __fastcall sub_2BE7B90(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // eax
  unsigned int v6; // r13d
  int v7; // eax
  unsigned __int8 *v8; // rsi
  __int16 v9; // bx
  unsigned int v10; // edx
  unsigned int v11; // eax
  unsigned __int8 v13; // r12
  char *v14; // rsi
  unsigned int v15; // eax
  _BYTE *v16; // rdi
  __int64 v17; // r9
  __m128i *v18; // rdi
  unsigned __int8 v19; // cl
  unsigned int v20; // r15d
  int v21; // eax
  size_t v22; // rdx
  unsigned __int8 v23; // al
  unsigned __int8 *v24; // rsi
  unsigned __int8 v25; // al
  unsigned __int8 *v26; // rsi
  signed __int8 v27; // al
  char *v28; // rsi
  unsigned __int8 v29; // [rsp+Fh] [rbp-81h]
  unsigned int v30; // [rsp+1Ch] [rbp-74h] BYREF
  void *dest; // [rsp+20h] [rbp-70h] BYREF
  size_t v32; // [rsp+28h] [rbp-68h]
  __int64 v33; // [rsp+30h] [rbp-60h] BYREF
  void *src; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD v36[8]; // [rsp+50h] [rbp-40h] BYREF

  v5 = *(_DWORD *)(a1 + 152);
  if ( v5 == 11 )
  {
    if ( (unsigned __int8)sub_2BE0030(a1) )
      return 0;
    v5 = *(_DWORD *)(a1 + 152);
  }
  if ( v5 == 16 )
  {
    v6 = sub_2BE0030(a1);
    if ( (_BYTE)v6 )
    {
      sub_2BE6030(
        (__int64)&src,
        *(_QWORD **)(a3 + 104),
        *(unsigned __int8 **)(a1 + 272),
        (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)));
      if ( !n )
        goto LABEL_36;
      LOBYTE(dest) = *(_BYTE *)src;
      sub_2BE78D0(a3, (char *)&dest);
      if ( n == 1 )
      {
        v19 = *(_BYTE *)src;
        if ( *a2 )
        {
          v25 = a2[1];
          v26 = *(unsigned __int8 **)(a3 + 8);
          LOBYTE(dest) = a2[1];
          if ( v26 == *(unsigned __int8 **)(a3 + 16) )
          {
            v29 = v19;
            sub_17EB120(a3, v26, (char *)&dest);
            v19 = v29;
          }
          else
          {
            if ( v26 )
            {
              *v26 = v25;
              v26 = *(unsigned __int8 **)(a3 + 8);
            }
            *(_QWORD *)(a3 + 8) = v26 + 1;
          }
        }
        else
        {
          *a2 = 1;
        }
        a2[1] = v19;
      }
      else if ( *a2 )
      {
        LOBYTE(dest) = a2[1];
        sub_2BE78D0(a3, (char *)&dest);
        *a2 = 0;
      }
      sub_2240A30((unsigned __int64 *)&src);
      return v6;
    }
    v5 = *(_DWORD *)(a1 + 152);
  }
  if ( v5 == 17 )
  {
    v6 = sub_2BE0030(a1);
    if ( !(_BYTE)v6 )
    {
      v5 = *(_DWORD *)(a1 + 152);
      goto LABEL_4;
    }
    if ( *a2 )
    {
      LOBYTE(src) = a2[1];
      sub_2BE78D0(a3, (char *)&src);
      *a2 = 0;
    }
    sub_2BE6030(
      (__int64)&dest,
      *(_QWORD **)(a3 + 104),
      *(unsigned __int8 **)(a1 + 272),
      (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)));
    if ( !v32 )
LABEL_36:
      abort();
    sub_2BE6250((__int64)&src, *(_QWORD **)(a3 + 104), dest, (__int64)dest + v32);
    v16 = dest;
    if ( src == v36 )
    {
      v22 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = v36[0];
        else
          memcpy(dest, src, n);
        v22 = n;
        v16 = dest;
      }
      v32 = v22;
      v16[v22] = 0;
      v16 = src;
      goto LABEL_59;
    }
    if ( dest == &v33 )
    {
      dest = src;
      v32 = n;
      v33 = v36[0];
    }
    else
    {
      v17 = v33;
      dest = src;
      v32 = n;
      v33 = v36[0];
      if ( v16 )
      {
        src = v16;
        v36[0] = v17;
LABEL_59:
        n = 0;
        *v16 = 0;
        sub_2240A30((unsigned __int64 *)&src);
        v18 = *(__m128i **)(a3 + 32);
        if ( v18 == *(__m128i **)(a3 + 40) )
        {
          sub_8FD760((__m128i **)(a3 + 24), v18, (__int64)&dest);
        }
        else
        {
          if ( v18 )
          {
            v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
            sub_2BDC2F0(v18->m128i_i64, dest, (__int64)dest + v32);
            v18 = *(__m128i **)(a3 + 32);
          }
          *(_QWORD *)(a3 + 32) = v18 + 2;
        }
        sub_2240A30((unsigned __int64 *)&dest);
        return v6;
      }
    }
    src = v36;
    v16 = v36;
    goto LABEL_59;
  }
LABEL_4:
  if ( v5 == 15 )
  {
    v6 = sub_2BE0030(a1);
    if ( (_BYTE)v6 )
    {
      if ( *a2 )
      {
        LOBYTE(src) = a2[1];
        sub_2BE78D0(a3, (char *)&src);
        *a2 = 0;
      }
      v11 = sub_2BE10D0(
              *(_QWORD **)(a3 + 104),
              *(_QWORD *)(a1 + 272),
              (char *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)),
              0);
      v10 = HIWORD(v11);
      if ( (v11 & 0x10000) == 0 && !(_WORD)v11 )
        goto LABEL_36;
      goto LABEL_15;
    }
  }
  v6 = sub_2BE0770(a1);
  if ( (_BYTE)v6 )
  {
    v13 = **(_BYTE **)(a1 + 272);
    if ( *a2 )
    {
      LOBYTE(src) = a2[1];
      sub_2BE78D0(a3, (char *)&src);
    }
    else
    {
      *a2 = 1;
    }
    a2[1] = v13;
    return v6;
  }
  v7 = *(_DWORD *)(a1 + 152);
  if ( v7 == 28 )
  {
    v15 = sub_2BE0030(a1);
    if ( !(_BYTE)v15 )
    {
      v7 = *(_DWORD *)(a1 + 152);
      goto LABEL_7;
    }
    v6 = *a2;
    if ( !(_BYTE)v6 )
    {
      if ( (*(_BYTE *)a1 & 0x10) != 0 )
      {
        v6 = v15;
        *(_WORD *)a2 = 11521;
        return v6;
      }
      if ( *(_DWORD *)(a1 + 152) != 11 || !(unsigned __int8)sub_2BE0030(a1) )
        goto LABEL_36;
      goto LABEL_42;
    }
    v20 = sub_2BE0770(a1);
    if ( (_BYTE)v20 )
    {
      v27 = **(_BYTE **)(a1 + 272);
      if ( v27 < (char)a2[1] )
        goto LABEL_36;
      LOBYTE(src) = a2[1];
      v28 = *(char **)(a3 + 56);
      BYTE1(src) = v27;
      if ( v28 == *(char **)(a3 + 64) )
      {
LABEL_101:
        sub_2BE7A20((unsigned __int64 *)(a3 + 48), v28, &src);
LABEL_94:
        *a2 = 0;
        return v20;
      }
    }
    else
    {
      v21 = *(_DWORD *)(a1 + 152);
      if ( v21 != 28 )
        goto LABEL_70;
      v20 = sub_2BE0030(a1);
      if ( !(_BYTE)v20 )
      {
        v21 = *(_DWORD *)(a1 + 152);
LABEL_70:
        if ( v21 != 11 )
          goto LABEL_36;
LABEL_42:
        if ( *a2 )
        {
          v23 = a2[1];
          v24 = *(unsigned __int8 **)(a3 + 8);
          LOBYTE(src) = a2[1];
          if ( v24 == *(unsigned __int8 **)(a3 + 16) )
          {
            sub_17EB120(a3, v24, (char *)&src);
          }
          else
          {
            if ( v24 )
              *v24 = v23;
            ++*(_QWORD *)(a3 + 8);
          }
        }
        else
        {
          *a2 = 1;
        }
        a2[1] = 45;
        return v6;
      }
      if ( (char)a2[1] > 45 )
        goto LABEL_36;
      LOBYTE(src) = a2[1];
      v28 = *(char **)(a3 + 56);
      BYTE1(src) = 45;
      if ( v28 == *(char **)(a3 + 64) )
        goto LABEL_101;
    }
    if ( v28 )
      *(_WORD *)v28 = (_WORD)src;
    *(_QWORD *)(a3 + 56) += 2LL;
    goto LABEL_94;
  }
LABEL_7:
  if ( v7 != 14 )
    goto LABEL_36;
  v6 = sub_2BE0030(a1);
  if ( !(_BYTE)v6 )
    goto LABEL_36;
  if ( *a2 )
  {
    LOBYTE(src) = a2[1];
    sub_2BE78D0(a3, (char *)&src);
    *a2 = 0;
  }
  v8 = *(unsigned __int8 **)(a1 + 272);
  v9 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 392) + 48LL) + 2LL * *v8) & 0x100;
  v30 = sub_2BE10D0(*(_QWORD **)(a3 + 104), (__int64)v8, (char *)&v8[*(_QWORD *)(a1 + 280)], 0);
  v10 = HIWORD(v30);
  if ( (v30 & 0x10000) == 0 && !(_WORD)v30 )
    goto LABEL_36;
  if ( !v9 )
  {
    LOWORD(v11) = v30;
LABEL_15:
    *(_WORD *)(a3 + 96) |= v11;
    *(_BYTE *)(a3 + 98) |= v10;
    return v6;
  }
  v14 = *(char **)(a3 + 80);
  if ( v14 == *(char **)(a3 + 88) )
  {
    sub_2BE3600((unsigned __int64 *)(a3 + 72), v14, &v30);
  }
  else
  {
    if ( v14 )
    {
      *(_DWORD *)v14 = v30;
      v14 = *(char **)(a3 + 80);
    }
    *(_QWORD *)(a3 + 80) = v14 + 4;
  }
  return v6;
}

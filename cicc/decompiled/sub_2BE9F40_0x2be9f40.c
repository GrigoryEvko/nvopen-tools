// Function: sub_2BE9F40
// Address: 0x2be9f40
//
__int64 __fastcall sub_2BE9F40(__int64 a1, _BYTE *a2, __int64 a3)
{
  int v4; // eax
  unsigned int v5; // r13d
  int v6; // eax
  unsigned __int8 *v7; // rsi
  __int16 v8; // bx
  unsigned int v9; // edx
  unsigned int v10; // eax
  char v12; // bl
  char *v13; // rsi
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // rax
  _BYTE *v18; // rdi
  __int64 v19; // r9
  __m128i *v20; // rdi
  unsigned int v21; // r15d
  int v22; // eax
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // r15d
  __int64 v26; // rax
  size_t v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // eax
  unsigned int v30; // [rsp+Ch] [rbp-94h]
  unsigned int v31; // [rsp+1Ch] [rbp-84h] BYREF
  _BYTE *v32; // [rsp+20h] [rbp-80h] BYREF
  __int64 v33; // [rsp+28h] [rbp-78h]
  void *dest; // [rsp+30h] [rbp-70h] BYREF
  size_t v35; // [rsp+38h] [rbp-68h]
  __int64 v36; // [rsp+40h] [rbp-60h] BYREF
  void *src; // [rsp+50h] [rbp-50h] BYREF
  size_t n; // [rsp+58h] [rbp-48h]
  _QWORD v39[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 152);
  if ( v4 == 11 )
  {
    if ( (unsigned __int8)sub_2BE0030(a1) )
      return 0;
    v4 = *(_DWORD *)(a1 + 152);
  }
  v32 = a2;
  v33 = a3;
  if ( v4 == 16 )
  {
    v5 = sub_2BE0030(a1);
    if ( (_BYTE)v5 )
    {
      sub_2BE6030(
        (__int64)&src,
        *(_QWORD **)(a3 + 112),
        *(unsigned __int8 **)(a1 + 272),
        (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)));
      if ( !n )
        goto LABEL_36;
      LOBYTE(dest) = sub_2BE35D0(*(_QWORD **)(a3 + 104), (unsigned int)*(char *)src);
      sub_2BE78D0(a3, (char *)&dest);
      if ( n == 1 )
      {
        sub_2BE7990(&v32, (unsigned int)*(char *)src);
      }
      else if ( *a2 )
      {
        v30 = (char)a2[1];
        v28 = sub_222F790(*(_QWORD **)(a3 + 104), v30);
        LOBYTE(dest) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v28 + 32LL))(v28, v30);
        sub_2BE78D0(a3, (char *)&dest);
        *a2 = 0;
      }
      sub_2240A30((unsigned __int64 *)&src);
      return v5;
    }
    v4 = *(_DWORD *)(a1 + 152);
  }
  if ( v4 == 17 )
  {
    v5 = sub_2BE0030(a1);
    if ( !(_BYTE)v5 )
    {
      v4 = *(_DWORD *)(a1 + 152);
      goto LABEL_4;
    }
    if ( *a2 )
    {
      v23 = (char)a2[1];
      v24 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
      LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v24 + 32LL))(v24, v23);
      sub_2BE78D0(a3, (char *)&src);
      *a2 = 0;
    }
    sub_2BE6030(
      (__int64)&dest,
      *(_QWORD **)(a3 + 112),
      *(unsigned __int8 **)(a1 + 272),
      (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)));
    if ( !v35 )
      goto LABEL_36;
    sub_2BE6250((__int64)&src, *(_QWORD **)(a3 + 112), dest, (__int64)dest + v35);
    v18 = dest;
    if ( src == v39 )
    {
      v27 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = v39[0];
        else
          memcpy(dest, src, n);
        v27 = n;
        v18 = dest;
      }
      v35 = v27;
      v18[v27] = 0;
      v18 = src;
      goto LABEL_57;
    }
    if ( dest == &v36 )
    {
      dest = src;
      v35 = n;
      v36 = v39[0];
    }
    else
    {
      v19 = v36;
      dest = src;
      v35 = n;
      v36 = v39[0];
      if ( v18 )
      {
        src = v18;
        v39[0] = v19;
LABEL_57:
        n = 0;
        *v18 = 0;
        sub_2240A30((unsigned __int64 *)&src);
        v20 = *(__m128i **)(a3 + 32);
        if ( v20 == *(__m128i **)(a3 + 40) )
        {
          sub_8FD760((__m128i **)(a3 + 24), v20, (__int64)&dest);
        }
        else
        {
          if ( v20 )
          {
            v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
            sub_2BDC2F0(v20->m128i_i64, dest, (__int64)dest + v35);
            v20 = *(__m128i **)(a3 + 32);
          }
          *(_QWORD *)(a3 + 32) = v20 + 2;
        }
        sub_2240A30((unsigned __int64 *)&dest);
        return v5;
      }
    }
    src = v39;
    v18 = v39;
    goto LABEL_57;
  }
LABEL_4:
  if ( v4 == 15 )
  {
    v5 = sub_2BE0030(a1);
    if ( (_BYTE)v5 )
    {
      if ( *a2 )
      {
        v25 = (char)a2[1];
        v26 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
        LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v26 + 32LL))(v26, v25);
        sub_2BE78D0(a3, (char *)&src);
        *a2 = 0;
      }
      v10 = sub_2BE10D0(
              *(_QWORD **)(a3 + 112),
              *(_QWORD *)(a1 + 272),
              (char *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)),
              1);
      v9 = HIWORD(v10);
      if ( (v10 & 0x10000) == 0 && !(_WORD)v10 )
        goto LABEL_36;
      goto LABEL_15;
    }
  }
  v5 = sub_2BE0770(a1);
  if ( (_BYTE)v5 )
  {
    v12 = **(_BYTE **)(a1 + 272);
    if ( *v32 )
    {
      v16 = (char)v32[1];
      v17 = sub_222F790(*(_QWORD **)(v33 + 104), (__int64)a2);
      LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v17 + 32LL))(v17, v16);
      sub_2BE78D0(v33, (char *)&src);
    }
    else
    {
      *v32 = 1;
    }
    v32[1] = v12;
    return v5;
  }
  v6 = *(_DWORD *)(a1 + 152);
  if ( v6 == 28 )
  {
    v5 = sub_2BE0030(a1);
    if ( (_BYTE)v5 )
    {
      if ( *a2 )
      {
        v21 = sub_2BE0770(a1);
        if ( (_BYTE)v21 )
        {
          v5 = v21;
          sub_2BE9AB0(a3, a2[1], **(_BYTE **)(a1 + 272));
          *a2 = 0;
          return v5;
        }
        v22 = *(_DWORD *)(a1 + 152);
        if ( v22 == 28 )
        {
          v29 = sub_2BE0030(a1);
          if ( (_BYTE)v29 )
          {
            v5 = v29;
            sub_2BE9AB0(a3, a2[1], 45);
            *a2 = 0;
            return v5;
          }
          v22 = *(_DWORD *)(a1 + 152);
        }
        if ( v22 != 11 )
          goto LABEL_36;
      }
      else if ( (*(_BYTE *)a1 & 0x10) == 0 )
      {
        if ( *(_DWORD *)(a1 + 152) != 11 || !(unsigned __int8)sub_2BE0030(a1) )
          goto LABEL_36;
        v5 = 0;
        sub_2BE7990(&v32, 45);
        return v5;
      }
      sub_2BE7990(&v32, 45);
      return v5;
    }
    v6 = *(_DWORD *)(a1 + 152);
  }
  if ( v6 != 14 )
    goto LABEL_36;
  v5 = sub_2BE0030(a1);
  if ( !(_BYTE)v5 )
    goto LABEL_36;
  if ( *a2 )
  {
    v14 = (char)a2[1];
    v15 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
    LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v15 + 32LL))(v15, v14);
    sub_2BE78D0(a3, (char *)&src);
    *a2 = 0;
  }
  v7 = *(unsigned __int8 **)(a1 + 272);
  v8 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 392) + 48LL) + 2LL * *v7) & 0x100;
  v31 = sub_2BE10D0(*(_QWORD **)(a3 + 112), (__int64)v7, (char *)&v7[*(_QWORD *)(a1 + 280)], 1);
  v9 = HIWORD(v31);
  if ( (v31 & 0x10000) == 0 && !(_WORD)v31 )
LABEL_36:
    abort();
  if ( !v8 )
  {
    LOWORD(v10) = v31;
LABEL_15:
    *(_WORD *)(a3 + 96) |= v10;
    *(_BYTE *)(a3 + 98) |= v9;
    return v5;
  }
  v13 = *(char **)(a3 + 80);
  if ( v13 == *(char **)(a3 + 88) )
  {
    sub_2BE3600((unsigned __int64 *)(a3 + 72), v13, &v31);
  }
  else
  {
    if ( v13 )
    {
      *(_DWORD *)v13 = v31;
      v13 = *(char **)(a3 + 80);
    }
    *(_QWORD *)(a3 + 80) = v13 + 4;
  }
  return v5;
}

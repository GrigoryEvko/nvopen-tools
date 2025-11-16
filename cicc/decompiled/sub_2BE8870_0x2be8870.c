// Function: sub_2BE8870
// Address: 0x2be8870
//
__int64 __fastcall sub_2BE8870(__int64 a1, _BYTE *a2, __int64 a3)
{
  int v5; // eax
  unsigned int v6; // r13d
  int v7; // eax
  unsigned __int8 *v8; // rsi
  __int16 v9; // bx
  unsigned int v10; // edx
  unsigned int v11; // eax
  char v13; // bl
  char *v14; // rsi
  unsigned int v15; // r15d
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // rax
  _BYTE *v19; // rdi
  __int64 v20; // r9
  __m128i *v21; // rdi
  unsigned int v22; // r15d
  int v23; // eax
  unsigned int v24; // r15d
  __int64 v25; // rax
  unsigned int v26; // r15d
  __int64 v27; // rax
  size_t v28; // rdx
  __int64 v29; // rax
  char v30; // al
  char *v31; // rsi
  unsigned int v32; // [rsp+Ch] [rbp-94h]
  unsigned int v33; // [rsp+1Ch] [rbp-84h] BYREF
  _BYTE *v34; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h]
  void *dest; // [rsp+30h] [rbp-70h] BYREF
  size_t v37; // [rsp+38h] [rbp-68h]
  __int64 v38; // [rsp+40h] [rbp-60h] BYREF
  void *src; // [rsp+50h] [rbp-50h] BYREF
  size_t n; // [rsp+58h] [rbp-48h]
  _QWORD v41[8]; // [rsp+60h] [rbp-40h] BYREF

  v5 = *(_DWORD *)(a1 + 152);
  if ( v5 == 11 )
  {
    if ( (unsigned __int8)sub_2BE0030(a1) )
      return 0;
    v5 = *(_DWORD *)(a1 + 152);
  }
  v34 = a2;
  v35 = a3;
  if ( v5 == 16 )
  {
    v6 = sub_2BE0030(a1);
    if ( (_BYTE)v6 )
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
        sub_2BE7900(&v34, (unsigned int)*(char *)src);
      }
      else if ( *a2 )
      {
        v32 = (char)a2[1];
        v29 = sub_222F790(*(_QWORD **)(a3 + 104), v32);
        LOBYTE(dest) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v29 + 32LL))(v29, v32);
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
      v24 = (char)a2[1];
      v25 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
      LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v25 + 32LL))(v25, v24);
      sub_2BE78D0(a3, (char *)&src);
      *a2 = 0;
    }
    sub_2BE6030(
      (__int64)&dest,
      *(_QWORD **)(a3 + 112),
      *(unsigned __int8 **)(a1 + 272),
      (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)));
    if ( !v37 )
LABEL_36:
      abort();
    sub_2BE6250((__int64)&src, *(_QWORD **)(a3 + 112), dest, (__int64)dest + v37);
    v19 = dest;
    if ( src == v41 )
    {
      v28 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = v41[0];
        else
          memcpy(dest, src, n);
        v28 = n;
        v19 = dest;
      }
      v37 = v28;
      v19[v28] = 0;
      v19 = src;
      goto LABEL_57;
    }
    if ( dest == &v38 )
    {
      dest = src;
      v37 = n;
      v38 = v41[0];
    }
    else
    {
      v20 = v38;
      dest = src;
      v37 = n;
      v38 = v41[0];
      if ( v19 )
      {
        src = v19;
        v41[0] = v20;
LABEL_57:
        n = 0;
        *v19 = 0;
        sub_2240A30((unsigned __int64 *)&src);
        v21 = *(__m128i **)(a3 + 32);
        if ( v21 == *(__m128i **)(a3 + 40) )
        {
          sub_8FD760((__m128i **)(a3 + 24), v21, (__int64)&dest);
        }
        else
        {
          if ( v21 )
          {
            v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
            sub_2BDC2F0(v21->m128i_i64, dest, (__int64)dest + v37);
            v21 = *(__m128i **)(a3 + 32);
          }
          *(_QWORD *)(a3 + 32) = v21 + 2;
        }
        sub_2240A30((unsigned __int64 *)&dest);
        return v6;
      }
    }
    src = v41;
    v19 = v41;
    goto LABEL_57;
  }
LABEL_4:
  if ( v5 == 15 )
  {
    v6 = sub_2BE0030(a1);
    if ( (_BYTE)v6 )
    {
      if ( *a2 )
      {
        v26 = (char)a2[1];
        v27 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
        LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v27 + 32LL))(v27, v26);
        sub_2BE78D0(a3, (char *)&src);
        *a2 = 0;
      }
      v11 = sub_2BE10D0(
              *(_QWORD **)(a3 + 112),
              *(_QWORD *)(a1 + 272),
              (char *)(*(_QWORD *)(a1 + 272) + *(_QWORD *)(a1 + 280)),
              1);
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
    if ( *v34 )
    {
      v17 = (char)v34[1];
      v18 = sub_222F790(*(_QWORD **)(v35 + 104), (__int64)a2);
      LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v18 + 32LL))(v18, v17);
      sub_2BE78D0(v35, (char *)&src);
    }
    else
    {
      *v34 = 1;
    }
    v34[1] = v13;
    return v6;
  }
  v7 = *(_DWORD *)(a1 + 152);
  if ( v7 == 28 )
  {
    v6 = sub_2BE0030(a1);
    if ( !(_BYTE)v6 )
    {
      v7 = *(_DWORD *)(a1 + 152);
      goto LABEL_7;
    }
    if ( !*a2 )
    {
      if ( (*(_BYTE *)a1 & 0x10) == 0 )
      {
        if ( *(_DWORD *)(a1 + 152) != 11 || !(unsigned __int8)sub_2BE0030(a1) )
          goto LABEL_36;
        v6 = 0;
        sub_2BE7900(&v34, 45);
        return v6;
      }
LABEL_66:
      sub_2BE7900(&v34, 45);
      return v6;
    }
    v22 = sub_2BE0770(a1);
    if ( (_BYTE)v22 )
    {
      v30 = **(_BYTE **)(a1 + 272);
      if ( v30 < (char)a2[1] )
        goto LABEL_36;
      LOBYTE(src) = a2[1];
      v31 = *(char **)(a3 + 56);
      BYTE1(src) = v30;
      if ( v31 == *(char **)(a3 + 64) )
      {
LABEL_87:
        sub_2BE7A20((unsigned __int64 *)(a3 + 48), v31, &src);
LABEL_81:
        *a2 = 0;
        return v22;
      }
    }
    else
    {
      v23 = *(_DWORD *)(a1 + 152);
      if ( v23 != 28 )
        goto LABEL_65;
      v22 = sub_2BE0030(a1);
      if ( !(_BYTE)v22 )
      {
        v23 = *(_DWORD *)(a1 + 152);
LABEL_65:
        if ( v23 != 11 )
          goto LABEL_36;
        goto LABEL_66;
      }
      if ( (char)a2[1] > 45 )
        goto LABEL_36;
      LOBYTE(src) = a2[1];
      v31 = *(char **)(a3 + 56);
      BYTE1(src) = 45;
      if ( v31 == *(char **)(a3 + 64) )
        goto LABEL_87;
    }
    if ( v31 )
      *(_WORD *)v31 = (_WORD)src;
    *(_QWORD *)(a3 + 56) += 2LL;
    goto LABEL_81;
  }
LABEL_7:
  if ( v7 != 14 )
    goto LABEL_36;
  v6 = sub_2BE0030(a1);
  if ( !(_BYTE)v6 )
    goto LABEL_36;
  if ( *a2 )
  {
    v15 = (char)a2[1];
    v16 = sub_222F790(*(_QWORD **)(a3 + 104), (__int64)a2);
    LOBYTE(src) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v16 + 32LL))(v16, v15);
    sub_2BE78D0(a3, (char *)&src);
    *a2 = 0;
  }
  v8 = *(unsigned __int8 **)(a1 + 272);
  v9 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 392) + 48LL) + 2LL * *v8) & 0x100;
  v33 = sub_2BE10D0(*(_QWORD **)(a3 + 112), (__int64)v8, (char *)&v8[*(_QWORD *)(a1 + 280)], 1);
  v10 = HIWORD(v33);
  if ( (v33 & 0x10000) == 0 && !(_WORD)v33 )
    goto LABEL_36;
  if ( !v9 )
  {
    LOWORD(v11) = v33;
LABEL_15:
    *(_WORD *)(a3 + 96) |= v11;
    *(_BYTE *)(a3 + 98) |= v10;
    return v6;
  }
  v14 = *(char **)(a3 + 80);
  if ( v14 == *(char **)(a3 + 88) )
  {
    sub_2BE3600((unsigned __int64 *)(a3 + 72), v14, &v33);
  }
  else
  {
    if ( v14 )
    {
      *(_DWORD *)v14 = v33;
      v14 = *(char **)(a3 + 80);
    }
    *(_QWORD *)(a3 + 80) = v14 + 4;
  }
  return v6;
}

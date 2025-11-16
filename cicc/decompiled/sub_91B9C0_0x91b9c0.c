// Function: sub_91B9C0
// Address: 0x91b9c0
//
__int64 *__fastcall sub_91B9C0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  unsigned int v5; // ecx
  __int64 (__fastcall ***v6)(); // rdi
  size_t v7; // rcx
  __int64 (__fastcall **v8)(); // rdx
  __int64 (__fastcall **v9)(); // rsi
  size_t v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  char v16; // al
  _QWORD *v17; // rsi
  _BYTE *v18; // rdi
  size_t v19; // rdx
  __int64 v20; // rcx
  __int64 (__fastcall ***v21)(); // [rsp+48h] [rbp-1E8h]
  _QWORD *v22; // [rsp+50h] [rbp-1E0h] BYREF
  size_t v23; // [rsp+58h] [rbp-1D8h]
  _QWORD v24[2]; // [rsp+60h] [rbp-1D0h] BYREF
  __int64 (__fastcall ***p_src)(); // [rsp+70h] [rbp-1C0h] BYREF
  size_t n; // [rsp+78h] [rbp-1B8h]
  __int64 (__fastcall **src)(); // [rsp+80h] [rbp-1B0h] BYREF
  _QWORD v28[3]; // [rsp+88h] [rbp-1A8h] BYREF
  unsigned __int64 v29; // [rsp+A0h] [rbp-190h]
  __int64 v30; // [rsp+A8h] [rbp-188h]
  unsigned __int64 v31; // [rsp+B0h] [rbp-180h]
  __int64 v32; // [rsp+B8h] [rbp-178h]
  _BYTE v33[8]; // [rsp+C0h] [rbp-170h] BYREF
  int v34; // [rsp+C8h] [rbp-168h]
  _QWORD v35[2]; // [rsp+D0h] [rbp-160h] BYREF
  _QWORD v36[2]; // [rsp+E0h] [rbp-150h] BYREF
  _QWORD v37[28]; // [rsp+F0h] [rbp-140h] BYREF
  __int16 v38; // [rsp+1D0h] [rbp-60h]
  __int64 v39; // [rsp+1D8h] [rbp-58h]
  __int64 v40; // [rsp+1E0h] [rbp-50h]
  __int64 v41; // [rsp+1E8h] [rbp-48h]
  __int64 v42; // [rsp+1F0h] [rbp-40h]

  *a1 = (__int64)(a1 + 2);
  v21 = (__int64 (__fastcall ***)())(a1 + 2);
  sub_91AB30(a1, (_BYTE *)*a2, *a2 + a2[1]);
  if ( (*(_BYTE *)(a3 + 156) & 8) == 0 )
  {
    LOBYTE(v5) = 0;
    if ( (*(_BYTE *)(a3 + 157) & 1) != 0 && *(_BYTE *)(a3 + 136) == 2 )
      v5 = (unsigned int)*(char *)(a3 + 173) >> 31;
    sub_91AF80((__int64)&p_src, a2, a3, v5);
    v6 = (__int64 (__fastcall ***)())*a1;
    if ( p_src == &src )
    {
      v11 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v6 = (_BYTE)src;
        else
          memcpy(v6, &src, n);
        v11 = n;
        v6 = (__int64 (__fastcall ***)())*a1;
      }
      a1[1] = v11;
      *((_BYTE *)v6 + v11) = 0;
      v6 = p_src;
      goto LABEL_9;
    }
    v7 = n;
    v8 = src;
    if ( v21 == v6 )
    {
      *a1 = (__int64)p_src;
      a1[1] = v7;
      a1[2] = (__int64)v8;
    }
    else
    {
      v9 = (__int64 (__fastcall **)())a1[2];
      *a1 = (__int64)p_src;
      a1[1] = v7;
      a1[2] = (__int64)v8;
      if ( v6 )
      {
        p_src = v6;
        src = v9;
        goto LABEL_9;
      }
    }
    p_src = &src;
    v6 = &src;
LABEL_9:
    n = 0;
    *(_BYTE *)v6 = 0;
    if ( p_src != &src )
      j_j___libc_free_0(p_src, (char *)src + 1);
    return a1;
  }
  sub_222DF20(v37);
  v37[27] = 0;
  v39 = 0;
  v40 = 0;
  v37[0] = off_4A06798;
  p_src = (__int64 (__fastcall ***)())qword_4A072D8;
  v38 = 0;
  v41 = 0;
  v42 = 0;
  *(__int64 (__fastcall ****)())((char *)&p_src + qword_4A072D8[-3]) = (__int64 (__fastcall ***)())&unk_4A07300;
  n = 0;
  sub_222DD70((char *)&p_src + (_QWORD)*(p_src - 3), 0);
  src = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v28[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  sub_222DD70((char *)&v28[-1] + (_QWORD)*(src - 3), 0);
  p_src = (__int64 (__fastcall ***)())qword_4A07328;
  *(__int64 (__fastcall ****)())((char *)&p_src + qword_4A07328[-3]) = (__int64 (__fastcall ***)())&unk_4A07378;
  p_src = (__int64 (__fastcall ***)())off_4A073F0;
  v37[0] = off_4A07440;
  src = off_4A07418;
  v28[0] = off_4A07480;
  v28[1] = 0;
  v28[2] = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  sub_220A990(v33);
  v35[0] = v36;
  v28[0] = off_4A07080;
  v34 = 24;
  v35[1] = 0;
  LOBYTE(v36[0]) = 0;
  sub_222DD70(v37, v28);
  sub_223E0D0(&src, "__cuda_local_var_", 17);
  v12 = sub_223E760(&src, *(unsigned int *)(a3 + 64));
  LOBYTE(v22) = 95;
  v13 = sub_223E0D0(v12, &v22, 1);
  v14 = sub_223E760(v13, *(unsigned __int16 *)(a3 + 68));
  LOBYTE(v22) = 95;
  sub_223E0D0(v14, &v22, 1);
  v15 = *(_QWORD *)(a3 + 120);
  if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8
    && (sub_8D4C10(v15, dword_4F077C4 != 2) & 1) != 0
    && ((v16 = *(_BYTE *)(a3 + 177), v16 == 3)
     || v16 == 2 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a3 + 184) + 48LL) - 1) <= 1u) )
  {
    sub_223E0D0(&src, "const_", 6);
  }
  else
  {
    sub_223E0D0(&src, "non_const_", 10);
  }
  sub_223E0D0(&src, *a2, a2[1]);
  v22 = v24;
  v23 = 0;
  LOBYTE(v24[0]) = 0;
  if ( v31 )
  {
    v17 = 0;
    if ( v31 > v29 )
      sub_2241130(&v22, 0, 0, v30, v31 - v30);
    else
      sub_2241130(&v22, 0, 0, v30, v29 - v30);
  }
  else
  {
    v17 = v35;
    sub_2240AE0(&v22, v35);
  }
  v18 = (_BYTE *)*a1;
  v19 = v23;
  if ( v22 == v24 )
  {
    if ( v23 )
    {
      if ( v23 == 1 )
      {
        *v18 = v24[0];
      }
      else
      {
        v17 = v24;
        memcpy(v18, v24, v23);
      }
      v19 = v23;
      v18 = (_BYTE *)*a1;
    }
    a1[1] = v19;
    v18[v19] = 0;
    v18 = v22;
    goto LABEL_29;
  }
  v20 = v24[0];
  if ( v21 == (__int64 (__fastcall ***)())v18 )
  {
    *a1 = (__int64)v22;
    a1[1] = v19;
    a1[2] = v20;
    goto LABEL_43;
  }
  v17 = (_QWORD *)a1[2];
  *a1 = (__int64)v22;
  a1[1] = v19;
  a1[2] = v20;
  if ( !v18 )
  {
LABEL_43:
    v22 = v24;
    v18 = v24;
    goto LABEL_29;
  }
  v22 = v18;
  v24[0] = v17;
LABEL_29:
  v23 = 0;
  *v18 = 0;
  if ( v22 != v24 )
  {
    v17 = (_QWORD *)(v24[0] + 1LL);
    j_j___libc_free_0(v22, v24[0] + 1LL);
  }
  p_src = (__int64 (__fastcall ***)())off_4A073F0;
  v37[0] = off_4A07440;
  src = off_4A07418;
  v28[0] = off_4A07080;
  if ( (_QWORD *)v35[0] != v36 )
  {
    v17 = (_QWORD *)(v36[0] + 1LL);
    j_j___libc_free_0(v35[0], v36[0] + 1LL);
  }
  v28[0] = off_4A07480;
  sub_2209150(v33, v17, v19);
  p_src = (__int64 (__fastcall ***)())qword_4A07328;
  *(__int64 (__fastcall ****)())((char *)&p_src + qword_4A07328[-3]) = (__int64 (__fastcall ***)())&unk_4A07378;
  src = (__int64 (__fastcall **)())qword_4A07288;
  *(_QWORD *)((char *)&v28[-1] + qword_4A07288[-3]) = &unk_4A072B0;
  p_src = (__int64 (__fastcall ***)())qword_4A072D8;
  *(__int64 (__fastcall ****)())((char *)&p_src + qword_4A072D8[-3]) = (__int64 (__fastcall ***)())&unk_4A07300;
  n = 0;
  v37[0] = off_4A06798;
  sub_222E050(v37);
  return a1;
}

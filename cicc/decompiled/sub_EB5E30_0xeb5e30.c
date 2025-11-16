// Function: sub_EB5E30
// Address: 0xeb5e30
//
__int64 __fastcall sub_EB5E30(__int64 a1, char a2)
{
  char *v3; // rsi
  bool v4; // zf
  unsigned int v5; // ebx
  __int64 v7; // rax
  unsigned int *v8; // r14
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  int v12; // ecx
  unsigned __int64 v13; // r13
  __m128i v14; // xmm0
  bool v15; // cc
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rcx
  unsigned int *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  size_t v27; // r14
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  size_t v30; // rax
  void *v31; // r13
  size_t v32; // r14
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  size_t v35; // r10
  size_t v36; // rax
  size_t v37; // rax
  char v38; // al
  char v39; // dl
  __int64 v40; // [rsp+0h] [rbp-90h]
  size_t v41; // [rsp+0h] [rbp-90h]
  char v43; // [rsp+Fh] [rbp-81h]
  __int64 v44; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v47; // [rsp+28h] [rbp-68h]
  void *s2; // [rsp+30h] [rbp-60h] BYREF
  size_t n; // [rsp+38h] [rbp-58h]
  __int64 v50; // [rsp+48h] [rbp-48h]
  unsigned int v51; // [rsp+50h] [rbp-40h]

  v43 = a2;
  v3 = *(char **)(a1 + 328);
  if ( v3 == *(char **)(a1 + 336) )
  {
    sub_EA9230((char **)(a1 + 320), v3, (_QWORD *)(a1 + 308));
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)(a1 + 308);
      v3 = *(char **)(a1 + 328);
    }
    *(_QWORD *)(a1 + 328) = v3 + 8;
  }
  v4 = *(_BYTE *)(a1 + 313) == 0;
  *(_DWORD *)(a1 + 308) = 1;
  if ( v4 )
  {
    v7 = sub_ECD7B0(a1);
    v40 = sub_ECD6A0(v7);
    while ( 1 )
    {
      v8 = *(unsigned int **)(a1 + 48);
      v9 = *v8;
      if ( (unsigned int)v9 <= 0x1A )
      {
        v10 = 67109377;
        if ( _bittest64(&v10, v9) )
          break;
      }
      v11 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = 0;
      v12 = v11;
      v13 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v11 - 40) >> 3);
      if ( (unsigned __int64)(40 * v11) > 0x28 )
      {
        do
        {
          v14 = _mm_loadu_si128((const __m128i *)v8 + 3);
          v15 = v8[8] <= 0x40;
          *v8 = v8[10];
          *(__m128i *)(v8 + 2) = v14;
          if ( !v15 )
          {
            v16 = *((_QWORD *)v8 + 3);
            if ( v16 )
              j_j___libc_free_0_0(v16);
          }
          v17 = *((_QWORD *)v8 + 8);
          v8 += 10;
          *((_QWORD *)v8 - 2) = v17;
          LODWORD(v17) = v8[8];
          v8[8] = 0;
          *(v8 - 2) = v17;
          --v13;
        }
        while ( v13 );
        v12 = *(_DWORD *)(a1 + 56);
        v8 = *(unsigned int **)(a1 + 48);
      }
      v18 = (unsigned int)(v12 - 1);
      *(_DWORD *)(a1 + 56) = v18;
      v19 = &v8[10 * v18];
      if ( v19[8] > 0x40 )
      {
        v20 = *((_QWORD *)v19 + 3);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
      if ( !*(_DWORD *)(a1 + 56) )
      {
        sub_1097F60(&s2, a1 + 40);
        sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&s2, v21, v22, v23);
        if ( v51 > 0x40 )
        {
          if ( v50 )
            j_j___libc_free_0_0(v50);
        }
      }
    }
    v24 = sub_ECD7B0(a1);
    v25 = sub_ECD6A0(v24);
    LOWORD(v51) = 259;
    v44 = v40;
    v45 = v25 - v40;
    s2 = "expected comma";
    if ( (unsigned __int8)sub_ECE210(a1, 26, &s2) )
      return 1;
    v46 = sub_EABDC0(a1);
    v47 = v26;
    v5 = sub_ECE000(a1);
    if ( (_BYTE)v5 )
    {
      return 1;
    }
    else
    {
      v27 = 0;
      v28 = sub_C935B0(&v46, byte_3F15413, 6, 0);
      v29 = v47;
      if ( v28 < v47 )
      {
        v27 = v47 - v28;
        v29 = v28;
      }
      n = v27;
      s2 = (void *)(v29 + v46);
      v30 = sub_C93740((__int64 *)&s2, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      v31 = s2;
      if ( v30 > n )
        v30 = n;
      v32 = v30 + n - v27;
      if ( v32 > n )
        v32 = n;
      v33 = sub_C935B0(&v44, byte_3F15413, 6, 0);
      v34 = v45;
      v35 = 0;
      if ( v33 < v45 )
      {
        v35 = v45 - v33;
        v34 = v33;
      }
      n = v35;
      s2 = (void *)(v34 + v44);
      v41 = v35;
      v36 = sub_C93740((__int64 *)&s2, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      if ( v36 > n )
        v36 = n;
      v37 = n - v41 + v36;
      if ( v37 > n )
        v37 = n;
      if ( v37 == v32 )
      {
        if ( v32 )
        {
          v39 = memcmp(s2, v31, v32) == 0;
          v38 = a2 == v39;
          v43 = a2 ^ v39;
        }
        else
        {
          v38 = a2;
          v43 = a2 ^ 1;
        }
      }
      else
      {
        v38 = a2 ^ 1;
      }
      *(_BYTE *)(a1 + 312) = v38;
      *(_BYTE *)(a1 + 313) = v43;
    }
  }
  else
  {
    sub_EB4E00(a1);
    return 0;
  }
  return v5;
}

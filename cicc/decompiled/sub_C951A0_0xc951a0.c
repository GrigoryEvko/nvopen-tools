// Function: sub_C951A0
// Address: 0xc951a0
//
__int64 sub_C951A0()
{
  __int64 v0; // rax
  const __m128i *v1; // r12
  __m128i v2; // xmm0
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rbx
  __int64 v10; // r15
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  int v17; // eax
  unsigned int v18; // r12d
  __m128i v20; // xmm1
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __cpu_mask v24; // rax
  int v25; // ecx
  unsigned int v26; // r14d
  __int64 v27; // r15
  __int64 v28; // rax
  void *v29; // rdx
  __int64 v30; // r12
  void *v31; // rdx
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rax
  int v35; // [rsp+18h] [rbp-258h]
  int v36; // [rsp+1Ch] [rbp-254h]
  int v37; // [rsp+20h] [rbp-250h]
  int v38; // [rsp+24h] [rbp-24Ch]
  const __m128i *v39; // [rsp+40h] [rbp-230h]
  unsigned __int64 v40; // [rsp+48h] [rbp-228h]
  unsigned __int64 v41; // [rsp+48h] [rbp-228h]
  __m128i v42; // [rsp+50h] [rbp-220h] BYREF
  __int64 v43; // [rsp+60h] [rbp-210h] BYREF
  unsigned __int64 v44; // [rsp+68h] [rbp-208h]
  _QWORD v45[2]; // [rsp+70h] [rbp-200h] BYREF
  char v46; // [rsp+80h] [rbp-1F0h]
  __m128i v47; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 v48; // [rsp+A0h] [rbp-1D0h] BYREF
  unsigned __int64 v49; // [rsp+A8h] [rbp-1C8h]
  cpu_set_t cpuset; // [rsp+B0h] [rbp-1C0h] BYREF
  cpu_set_t v51; // [rsp+130h] [rbp-140h] BYREF
  const char *v52; // [rsp+1B0h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+1B8h] [rbp-B8h]
  _QWORD v54[2]; // [rsp+1C0h] [rbp-B0h] BYREF
  char v55; // [rsp+1D0h] [rbp-A0h]
  char v56; // [rsp+1D1h] [rbp-9Fh]

  if ( sched_getaffinity(0, 0x80u, &cpuset) )
    return (unsigned int)-1;
  v56 = 1;
  memset(&v51, 0, sizeof(v51));
  v52 = "/proc/cpuinfo";
  v55 = 3;
  sub_C7E060((__int64)v45, (__int64 *)&v52);
  if ( (v46 & 1) == 0 || (v26 = v45[0]) == 0 )
  {
    v52 = (const char *)v54;
    v53 = 0x800000000LL;
    v0 = *(_QWORD *)(v45[0] + 16LL) - *(_QWORD *)(v45[0] + 8LL);
    v47.m128i_i64[0] = *(_QWORD *)(v45[0] + 8LL);
    v47.m128i_i64[1] = v0;
    sub_C937F0(&v47, (__int64)&v52, "\n", 1u, 0xFFFFFFFFLL, 0);
    v1 = (const __m128i *)v52;
    v39 = (const __m128i *)&v52[16 * (unsigned int)v53];
    if ( v39 == (const __m128i *)v52 )
    {
LABEL_33:
      v18 = __sched_cpucount(0x80u, &v51);
      if ( v52 != (const char *)v54 )
        _libc_free(v52, &v51);
      if ( (v46 & 1) == 0 )
        goto LABEL_73;
      return v18;
    }
    v35 = -1;
    v36 = -1;
    v37 = -1;
    v38 = -1;
    while ( 1 )
    {
      v2 = _mm_loadu_si128(v1);
      LOBYTE(v43) = 58;
      v42 = v2;
      v3 = sub_C931B0(v42.m128i_i64, &v43, 1u, 0);
      if ( v3 == -1 )
      {
        v20 = _mm_loadu_si128(&v42);
        v48 = 0;
        v49 = 0;
        v47 = v20;
      }
      else
      {
        v4 = v3 + 1;
        if ( v3 + 1 > v42.m128i_i64[1] )
        {
          v4 = v42.m128i_i64[1];
          v5 = 0;
        }
        else
        {
          v5 = v42.m128i_i64[1] - v4;
        }
        v47.m128i_i64[0] = v42.m128i_i64[0];
        if ( v3 > v42.m128i_i64[1] )
          v3 = v42.m128i_u64[1];
        v49 = v5;
        v48 = v4 + v42.m128i_i64[0];
        v47.m128i_i64[1] = v3;
      }
      v6 = sub_C935B0(&v47, byte_3F15413, 6, 0);
      v7 = v47.m128i_i64[1];
      v8 = 0;
      if ( v6 < v47.m128i_i64[1] )
      {
        v8 = v47.m128i_i64[1] - v6;
        v7 = v6;
      }
      v44 = v8;
      v40 = v8;
      v43 = v47.m128i_i64[0] + v7;
      v9 = sub_C93740(&v43, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      v10 = v43;
      if ( v9 > v44 )
        v9 = v44;
      v11 = v44 - v40 + v9;
      if ( v11 > v44 )
        v11 = v44;
      v12 = sub_C935B0(&v48, byte_3F15413, 6, 0);
      v13 = v49;
      v14 = 0;
      if ( v12 < v49 )
      {
        v14 = v49 - v12;
        v13 = v12;
      }
      v44 = v14;
      v41 = v14;
      v43 = v48 + v13;
      v15 = sub_C93740(&v43, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL) + 1;
      if ( v15 > v44 )
        v15 = v44;
      v16 = v15 + v44 - v41;
      if ( v16 > v44 )
        v16 = v44;
      if ( v11 == 9 )
      {
        if ( *(_QWORD *)v10 == 0x6F737365636F7270LL && *(_BYTE *)(v10 + 8) == 114 && !sub_C93CC0(v43, v16, 0xAu, &v43) )
        {
          v22 = v43;
          if ( v43 != (int)v43 )
            v22 = v38;
          v38 = v22;
        }
        goto LABEL_6;
      }
      if ( v11 == 11 )
      {
        if ( *(_QWORD *)v10 == 0x6C61636973796870LL
          && *(_WORD *)(v10 + 8) == 26912
          && *(_BYTE *)(v10 + 10) == 100
          && !sub_C93CC0(v43, v16, 0xAu, &v43) )
        {
          v21 = v43;
          if ( v43 != (int)v43 )
            v21 = v37;
          v37 = v21;
        }
        goto LABEL_6;
      }
      if ( v11 != 8 )
        break;
      if ( *(_QWORD *)v10 != 0x73676E696C626973LL || sub_C93CC0(v43, v16, 0xAu, &v43) )
      {
LABEL_6:
        if ( v39 == ++v1 )
          goto LABEL_33;
      }
      else
      {
        v17 = v43;
        if ( v43 != (int)v43 )
          v17 = v36;
        ++v1;
        v36 = v17;
        if ( v39 == v1 )
          goto LABEL_33;
      }
    }
    if ( v11 == 7 && *(_DWORD *)v10 == 1701998435 && *(_WORD *)(v10 + 4) == 26912 && *(_BYTE *)(v10 + 6) == 100 )
    {
      if ( !sub_C93CC0(v43, v16, 0xAu, &v43) )
      {
        v23 = v43;
        if ( v43 != (int)v43 )
          v23 = v35;
        v35 = v23;
      }
      if ( (unsigned __int64)v38 <= 0x3FF )
      {
        v24 = cpuset.__bits[(unsigned __int64)v38 >> 6];
        if ( _bittest64((const __int64 *)&v24, v38) )
        {
          v25 = v35 + v37 * v36;
          if ( (unsigned __int64)v25 <= 0x3FF )
            v51.__bits[(unsigned __int64)v25 >> 6] |= 1LL << v25;
        }
      }
    }
    goto LABEL_6;
  }
  v27 = v45[1];
  v28 = sub_CB72A0(v45, &v52);
  v29 = *(void **)(v28 + 32);
  v30 = v28;
  if ( *(_QWORD *)(v28 + 24) - (_QWORD)v29 <= 0xAu )
  {
    v34 = sub_CB6200(v28, "Can't read ", 11);
    v31 = *(void **)(v34 + 32);
    v30 = v34;
  }
  else
  {
    qmemcpy(v29, "Can't read ", 11);
    v31 = (void *)(*(_QWORD *)(v28 + 32) + 11LL);
    *(_QWORD *)(v28 + 32) = v31;
  }
  if ( *(_QWORD *)(v30 + 24) - (_QWORD)v31 <= 0xEu )
  {
    v30 = sub_CB6200(v30, "/proc/cpuinfo: ", 15);
  }
  else
  {
    qmemcpy(v31, "/proc/cpuinfo: ", 15);
    *(_QWORD *)(v30 + 32) += 15LL;
  }
  (*(void (__fastcall **)(const char **, __int64, _QWORD))(*(_QWORD *)v27 + 32LL))(&v52, v27, v26);
  v32 = sub_CB6200(v30, v52, v53);
  v33 = *(_BYTE **)(v32 + 32);
  if ( *(_BYTE **)(v32 + 24) == v33 )
  {
    sub_CB6200(v32, "\n", 1);
  }
  else
  {
    *v33 = 10;
    ++*(_QWORD *)(v32 + 32);
  }
  if ( v52 != (const char *)v54 )
    j_j___libc_free_0(v52, v54[0] + 1LL);
  v18 = -1;
  if ( (v46 & 1) == 0 )
  {
LABEL_73:
    if ( v45[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v45[0] + 8LL))(v45[0]);
  }
  return v18;
}

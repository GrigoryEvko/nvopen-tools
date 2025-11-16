// Function: sub_1AFD4E0
// Address: 0x1afd4e0
//
__int64 __fastcall sub_1AFD4E0(
        __int64 a1,
        char a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v15; // rax
  __int64 v16; // rax
  const void *v17; // r12
  __int64 result; // rax
  __int64 v19; // r8
  _QWORD *v20; // rbx
  __int64 v21; // r13
  _QWORD *v22; // r12
  __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdi
  int v29; // ecx
  __int64 v30; // r10
  int v31; // ecx
  unsigned int v32; // r9d
  __int64 *v33; // rdx
  __int64 v34; // r11
  _QWORD *v35; // rax
  unsigned int v36; // r9d
  __int64 *v37; // rdx
  __int64 v38; // r11
  _QWORD *v39; // rdx
  unsigned __int64 *v40; // rcx
  unsigned __int64 v41; // rdx
  double v42; // xmm4_8
  double v43; // xmm5_8
  int v44; // edx
  int v45; // r8d
  int v46; // edx
  int v47; // r8d
  __int32 v48; // eax
  __int64 v49; // rdi
  _QWORD *v50; // rcx
  __int64 v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rax
  __int64 v54; // r14
  bool v55; // al
  char *v56; // [rsp+0h] [rbp-220h]
  size_t n; // [rsp+8h] [rbp-218h]
  char *v59; // [rsp+20h] [rbp-200h]
  __int64 v62; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v63[2]; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 v64; // [rsp+50h] [rbp-1D0h]
  __m128i v65; // [rsp+60h] [rbp-1C0h] BYREF
  _QWORD v66[54]; // [rsp+70h] [rbp-1B0h] BYREF

  if ( a4 && a2 )
  {
    v65.m128i_i64[0] = (__int64)v66;
    v65.m128i_i64[1] = 0x1000000000LL;
    sub_1B67580(a1, a4, a5, a3, &v65);
    while ( 1 )
    {
      v48 = v65.m128i_i32[2];
      v49 = v65.m128i_i64[0];
      if ( !v65.m128i_i32[2] )
        break;
      v63[0] = 6;
      v63[1] = 0;
      v50 = (_QWORD *)(v65.m128i_i64[0] + 24LL * v65.m128i_u32[2] - 24);
      v64 = v50[2];
      if ( v64 != -8 && v64 != 0 && v64 != -16 )
      {
        sub_1649AC0(v63, *v50 & 0xFFFFFFFFFFFFFFF8LL);
        v48 = v65.m128i_i32[2];
        v49 = v65.m128i_i64[0];
      }
      v51 = (unsigned int)(v48 - 1);
      v65.m128i_i32[2] = v51;
      v52 = (_QWORD *)(v49 + 24 * v51);
      v53 = v52[2];
      if ( v53 != -8 && v53 != 0 && v53 != -16 )
        sub_1649B30(v52);
      v54 = v64;
      if ( v64 )
      {
        v55 = v64 != -8 && v64 != -16;
        if ( *(_BYTE *)(v64 + 16) <= 0x17u )
        {
          if ( v55 )
            sub_1649B30(v63);
        }
        else
        {
          if ( v55 )
            sub_1649B30(v63);
          sub_1AEB370(v54, 0);
        }
      }
    }
    if ( (_QWORD *)v65.m128i_i64[0] != v66 )
      _libc_free(v65.m128i_u64[0]);
  }
  v15 = sub_157EB90(**(_QWORD **)(a1 + 32));
  v16 = sub_1632FA0(v15);
  v17 = *(const void **)(a1 + 32);
  v62 = v16;
  n = *(_QWORD *)(a1 + 40) - (_QWORD)v17;
  result = 0x7FFFFFFFFFFFFFF8LL;
  if ( n > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( n )
  {
    v56 = (char *)sub_22077B0(n);
    memcpy(v56, v17, n);
    if ( v56 != &v56[n] )
    {
      v59 = v56;
      while ( 1 )
      {
        v20 = *(_QWORD **)(*(_QWORD *)v59 + 48LL);
        v21 = *(_QWORD *)v59 + 40LL;
        if ( (_QWORD *)v21 != v20 )
          break;
LABEL_22:
        v59 += 8;
        if ( &v56[n] == v59 )
          return j_j___libc_free_0(v56, n);
      }
      while ( 1 )
      {
        v22 = v20;
        v20 = (_QWORD *)v20[1];
        v65 = (__m128i)(unsigned __int64)v62;
        v66[2] = 0;
        v66[0] = a5;
        v66[1] = a6;
        v23 = sub_13E3350((__int64)(v22 - 3), &v65, 0, 0, v19);
        v26 = v23;
        if ( v23 )
        {
          if ( *(_BYTE *)(v23 + 16) > 0x17u )
          {
            v27 = *(_QWORD *)(v23 + 40);
            v28 = v22[2];
            if ( v27 != v28 )
            {
              v29 = *(_DWORD *)(a3 + 24);
              if ( v29 )
              {
                v30 = *(_QWORD *)(a3 + 8);
                v31 = v29 - 1;
                v32 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
                v33 = (__int64 *)(v30 + 16LL * v32);
                v34 = *v33;
                if ( v27 == *v33 )
                {
LABEL_13:
                  v35 = (_QWORD *)v33[1];
                  if ( v35 )
                  {
                    v36 = v31 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
                    v37 = (__int64 *)(v30 + 16LL * v36);
                    v38 = *v37;
                    if ( v28 != *v37 )
                    {
                      v46 = 1;
                      while ( v38 != -8 )
                      {
                        v47 = v46 + 1;
                        v36 = v31 & (v46 + v36);
                        v37 = (__int64 *)(v30 + 16LL * v36);
                        v38 = *v37;
                        if ( v28 == *v37 )
                          goto LABEL_15;
                        v46 = v47;
                      }
                      goto LABEL_19;
                    }
LABEL_15:
                    v39 = (_QWORD *)v37[1];
                    if ( v39 != v35 )
                    {
                      while ( v39 )
                      {
                        v39 = (_QWORD *)*v39;
                        if ( v39 == v35 )
                          goto LABEL_18;
                      }
                      goto LABEL_19;
                    }
                  }
                }
                else
                {
                  v44 = 1;
                  while ( v34 != -8 )
                  {
                    v45 = v44 + 1;
                    v32 = v31 & (v44 + v32);
                    v33 = (__int64 *)(v30 + 16LL * v32);
                    v34 = *v33;
                    if ( v27 == *v33 )
                      goto LABEL_13;
                    v44 = v45;
                  }
                }
              }
            }
          }
LABEL_18:
          sub_164D160((__int64)(v22 - 3), v26, a7, a8, a9, a10, v24, v25, a13, a14);
        }
LABEL_19:
        if ( sub_1AE9990((__int64)(v22 - 3), 0) )
        {
          sub_157EA20(v21, (__int64)(v22 - 3));
          v40 = (unsigned __int64 *)v22[1];
          v41 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
          *v40 = v41 | *v40 & 7;
          *(_QWORD *)(v41 + 8) = v40;
          *v22 &= 7uLL;
          v22[1] = 0;
          sub_164BEC0((__int64)(v22 - 3), (__int64)(v22 - 3), v41, (__int64)v40, a7, a8, a9, a10, v42, v43, a13, a14);
        }
        if ( (_QWORD *)v21 == v20 )
          goto LABEL_22;
      }
    }
    return j_j___libc_free_0(v56, n);
  }
  return result;
}

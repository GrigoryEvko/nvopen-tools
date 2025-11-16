// Function: sub_C27E00
// Address: 0xc27e00
//
__int64 __fastcall sub_C27E00(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v3; // r14
  __int64 result; // rax
  unsigned int v5; // r12d
  int v6; // edx
  unsigned int v7; // ebx
  unsigned __int64 v8; // r14
  __int64 v9; // rsi
  __int64 *v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // eax
  int v18; // edx
  _QWORD *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // r14
  int v22; // eax
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // r10
  _QWORD *v30; // r9
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // r11
  const void *v33; // rsi
  const void *v34; // rdi
  size_t v35; // rdx
  unsigned int v36; // eax
  unsigned int v37; // [rsp+14h] [rbp-1ACh]
  unsigned __int64 v38; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v39; // [rsp+28h] [rbp-198h]
  unsigned __int64 *v40; // [rsp+30h] [rbp-190h]
  _QWORD *v41; // [rsp+30h] [rbp-190h]
  __int64 v42; // [rsp+38h] [rbp-188h]
  unsigned __int64 *v43; // [rsp+38h] [rbp-188h]
  __int64 v44; // [rsp+48h] [rbp-178h]
  __int64 v45; // [rsp+50h] [rbp-170h]
  __int64 *v46; // [rsp+58h] [rbp-168h]
  unsigned int i; // [rsp+58h] [rbp-168h]
  _QWORD *v49; // [rsp+60h] [rbp-160h]
  unsigned __int64 *v50; // [rsp+68h] [rbp-158h]
  unsigned int v51; // [rsp+68h] [rbp-158h]
  _QWORD *v52; // [rsp+68h] [rbp-158h]
  bool v53; // [rsp+77h] [rbp-149h] BYREF
  _DWORD v54[2]; // [rsp+78h] [rbp-148h] BYREF
  __m128i v55; // [rsp+80h] [rbp-140h] BYREF
  unsigned __int64 v56; // [rsp+90h] [rbp-130h] BYREF
  char v57; // [rsp+A0h] [rbp-120h]
  unsigned int v58; // [rsp+B0h] [rbp-110h] BYREF
  char v59; // [rsp+C0h] [rbp-100h]
  unsigned int v60[8]; // [rsp+D0h] [rbp-F0h] BYREF
  unsigned int v61[8]; // [rsp+F0h] [rbp-D0h] BYREF
  unsigned __int64 v62; // [rsp+110h] [rbp-B0h] BYREF
  char v63; // [rsp+120h] [rbp-A0h]
  __int64 v64; // [rsp+130h] [rbp-90h] BYREF
  char v65; // [rsp+140h] [rbp-80h]
  __m128i *v66; // [rsp+150h] [rbp-70h] BYREF
  __int64 v67; // [rsp+158h] [rbp-68h]
  char v68; // [rsp+160h] [rbp-60h]
  __m128i v69; // [rsp+170h] [rbp-50h] BYREF
  char v70; // [rsp+180h] [rbp-40h]

  v3 = a2;
  sub_C21E40((__int64)&v56, (_QWORD *)a1);
  if ( (v57 & 1) != 0 )
  {
    result = (unsigned int)v56;
    if ( (_DWORD)v56 )
      return result;
  }
  a2[7] = sub_C1B1E0(v56, 1u, a2[7], (bool *)v69.m128i_i8);
  sub_C22200((__int64)&v58, (_QWORD *)a1);
  if ( (v59 & 1) != 0 )
  {
    result = v58;
    if ( v58 )
      return result;
  }
  if ( !v58 )
  {
    sub_C22200((__int64)&v62, (_QWORD *)a1);
    goto LABEL_58;
  }
  v37 = 0;
  v40 = a2 + 10;
  do
  {
    sub_C21E40((__int64)v60, (_QWORD *)a1);
    result = sub_C21E20(v60);
    if ( (_DWORD)result )
      return result;
    if ( (v60[0] & 0xFFFF0000) != 0 )
    {
      sub_2241E40();
      return 0;
    }
    sub_C21E40((__int64)v61, (_QWORD *)a1);
    result = sub_C21E20(v61);
    if ( (_DWORD)result )
      return result;
    sub_C21E40((__int64)&v62, (_QWORD *)a1);
    result = sub_C21E20(&v62);
    if ( (_DWORD)result )
      return result;
    sub_C22200((__int64)&v64, (_QWORD *)a1);
    if ( (v65 & 1) != 0 )
    {
      result = (unsigned int)v64;
      if ( (_DWORD)v64 )
        return result;
      v5 = v61[0];
      if ( !*(_BYTE *)(a1 + 184) || (v6 = *(_DWORD *)(a1 + 200), v6 == 31) )
      {
LABEL_83:
        v42 = (__int64)v40;
        goto LABEL_31;
      }
    }
    else
    {
      LODWORD(result) = v64;
      v5 = v61[0];
      if ( !*(_BYTE *)(a1 + 184) )
        goto LABEL_14;
      v6 = *(_DWORD *)(a1 + 200);
      if ( v6 == 31 )
        goto LABEL_14;
    }
    v5 &= ~(-1 << (v6 + 1));
LABEL_14:
    if ( !(_DWORD)result )
      goto LABEL_83;
    v7 = 0;
    do
    {
      sub_C21FD0((__int64)&v66, (_QWORD *)a1, 0);
      if ( (v68 & 1) != 0 )
      {
        result = (unsigned int)v66;
        if ( (_DWORD)v66 )
          return result;
      }
      sub_C21E40((__int64)&v69, (_QWORD *)a1);
      if ( (v70 & 1) != 0 )
      {
        result = v69.m128i_u32[0];
        if ( v69.m128i_i32[0] )
          return result;
      }
      v8 = v69.m128i_i64[0];
      v9 = (__int64)v40;
      v54[1] = v5;
      v10 = (__int64 *)v66;
      v11 = v67;
      v42 = (__int64)v40;
      v12 = a2[11];
      v54[0] = v60[0];
      if ( !v12 )
        goto LABEL_45;
      do
      {
        while ( 1 )
        {
          if ( v60[0] > *(_DWORD *)(v12 + 32) )
          {
            v12 = *(_QWORD *)(v12 + 24);
            goto LABEL_26;
          }
          if ( v60[0] == *(_DWORD *)(v12 + 32) && v5 > *(_DWORD *)(v12 + 36) )
            break;
          v9 = v12;
          v12 = *(_QWORD *)(v12 + 16);
          if ( !v12 )
            goto LABEL_27;
        }
        v12 = *(_QWORD *)(v12 + 24);
LABEL_26:
        ;
      }
      while ( v12 );
LABEL_27:
      if ( v40 == (unsigned __int64 *)v9
        || v60[0] < *(_DWORD *)(v9 + 32)
        || v60[0] == *(_DWORD *)(v9 + 32) && v5 < *(_DWORD *)(v9 + 36) )
      {
LABEL_45:
        v45 = v67;
        v55.m128i_i64[0] = (__int64)v54;
        v46 = (__int64 *)v66;
        v16 = sub_C272A0(a2 + 9, v9, (__int64 **)&v55);
        v11 = v45;
        v10 = v46;
        v9 = v16;
      }
      v55.m128i_i64[0] = (__int64)v10;
      ++v7;
      v55.m128i_i64[1] = v11;
      v50 = sub_C1CD30((_QWORD *)(v9 + 48), &v55);
      *v50 = sub_C1B1E0(v8, 1u, *v50, &v53);
    }
    while ( (unsigned int)v64 > v7 );
LABEL_31:
    v13 = v62;
    v14 = v42;
    v15 = a2[11];
    v69.m128i_i64[0] = __PAIR64__(v5, v60[0]);
    if ( !v15 )
      goto LABEL_55;
    do
    {
      if ( v60[0] > *(_DWORD *)(v15 + 32) || v60[0] == *(_DWORD *)(v15 + 32) && *(_DWORD *)(v15 + 36) < v5 )
      {
        v15 = *(_QWORD *)(v15 + 24);
      }
      else
      {
        v14 = v15;
        v15 = *(_QWORD *)(v15 + 16);
      }
    }
    while ( v15 );
    if ( v14 == v42 || v60[0] < *(_DWORD *)(v14 + 32) || v60[0] == *(_DWORD *)(v14 + 32) && *(_DWORD *)(v14 + 36) > v5 )
    {
LABEL_55:
      v66 = &v69;
      v14 = sub_C272A0(a2 + 9, v14, (__int64 **)&v66);
    }
    ++v37;
    *(_QWORD *)(v14 + 40) = sub_C1B1E0(v13, 1u, *(_QWORD *)(v14 + 40), (bool *)&v66);
  }
  while ( v58 > v37 );
  v3 = a2;
  sub_C22200((__int64)&v62, (_QWORD *)a1);
LABEL_58:
  if ( (v63 & 1) == 0 || (result = (unsigned int)v62, !(_DWORD)v62) )
  {
    if ( (_DWORD)v62 )
    {
      v43 = v3;
      for ( i = 0; (unsigned int)v62 > i; ++i )
      {
        sub_C21E40((__int64)&v64, (_QWORD *)a1);
        result = sub_C21E20(&v64);
        if ( (_DWORD)result )
          return result;
        sub_C21E40((__int64)&v66, (_QWORD *)a1);
        result = sub_C21E20(&v66);
        v51 = result;
        if ( (_DWORD)result )
          return result;
        sub_C21FD0((__int64)&v69, (_QWORD *)a1, 0);
        if ( (v70 & 1) != 0 )
        {
          result = v69.m128i_u32[0];
          if ( v69.m128i_i32[0] )
            return result;
        }
        v17 = (unsigned int)v66;
        if ( *(_BYTE *)(a1 + 184) )
        {
          v18 = *(_DWORD *)(a1 + 200);
          if ( v18 != 31 )
            v17 = (unsigned int)v66 & ~(-1 << (v18 + 1));
        }
        v61[1] = v17;
        v61[0] = v64;
        v19 = (_QWORD *)sub_C273A0(v43, v61);
        v20 = v19[2];
        v49 = v19 + 1;
        if ( v20 )
        {
          v21 = (__int64)(v19 + 1);
          do
          {
            if ( sub_C1F8C0(v20 + 32, (__int64)&v69) >= 0 )
            {
              v21 = v20;
              v20 = *(_QWORD *)(v20 + 16);
            }
            else
            {
              v20 = *(_QWORD *)(v20 + 24);
            }
          }
          while ( v20 );
          if ( (_QWORD *)v21 != v49 )
          {
            v22 = sub_C1F8C0((__int64)&v69, v21 + 32);
            v23 = v21 + 48;
            if ( v22 >= 0 )
              goto LABEL_80;
          }
        }
        else
        {
          v21 = (__int64)(v19 + 1);
        }
        v26 = (_QWORD *)v21;
        v21 = sub_22077B0(224);
        *(__m128i *)(v21 + 32) = _mm_loadu_si128(&v69);
        memset((void *)(v21 + 48), 0, 0xB0u);
        *(_QWORD *)(v21 + 144) = v21 + 128;
        *(_QWORD *)(v21 + 152) = v21 + 128;
        *(_QWORD *)(v21 + 192) = v21 + 176;
        *(_QWORD *)(v21 + 200) = v21 + 176;
        v27 = sub_C1C960(v19, v26, (const void **)(v21 + 32));
        v29 = v21 + 48;
        v30 = v28;
        if ( !v28 )
        {
          v52 = v27;
          sub_C1FCF0(v21 + 48);
          j_j___libc_free_0(v21, 224);
          v23 = (__int64)(v52 + 6);
          v21 = (__int64)v52;
          goto LABEL_80;
        }
        if ( v49 == v28 || v27 )
        {
LABEL_88:
          v51 = 1;
          goto LABEL_89;
        }
        v31 = v28[5];
        v32 = *(_QWORD *)(v21 + 40);
        v33 = (const void *)v28[4];
        v34 = *(const void **)(v21 + 32);
        if ( v31 < v32 )
        {
          if ( v33 == v34 )
          {
LABEL_101:
            v51 = v31 > v32;
            goto LABEL_89;
          }
          v35 = v28[5];
        }
        else
        {
          if ( v33 == v34 )
            goto LABEL_100;
          v35 = *(_QWORD *)(v21 + 40);
        }
        v38 = *(_QWORD *)(v21 + 40);
        v39 = v31;
        if ( !v34 )
          goto LABEL_88;
        if ( !v33 )
          goto LABEL_89;
        v41 = v30;
        v36 = memcmp(v34, v33, v35);
        v29 = v21 + 48;
        v30 = v41;
        v31 = v39;
        v32 = v38;
        if ( v36 )
        {
          v51 = v36 >> 31;
          goto LABEL_89;
        }
LABEL_100:
        if ( v31 != v32 )
          goto LABEL_101;
LABEL_89:
        v44 = v29;
        sub_220F040(v51, v21, v30, v49);
        ++v19[5];
        v23 = v44;
LABEL_80:
        v24 = v69.m128i_i64[1];
        v25 = v69.m128i_i64[0];
        *(_QWORD *)(v21 + 80) = 0;
        *(_QWORD *)(v21 + 64) = v25;
        *(_QWORD *)(v21 + 72) = v24;
        *(_QWORD *)(v21 + 88) = 0;
        *(_DWORD *)(v21 + 96) = 0;
        result = sub_C27E00(a1, v23);
        if ( (_DWORD)result )
          return result;
      }
    }
    sub_C1AFD0();
    return 0;
  }
  return result;
}

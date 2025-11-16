// Function: sub_1248090
// Address: 0x1248090
//
_BYTE *__fastcall sub_1248090(__int64 a1, _QWORD *a2, unsigned int a3, unsigned __int64 a4)
{
  unsigned __int32 v5; // r14d
  unsigned int v8; // r13d
  __int64 v9; // rdi
  unsigned __int64 *v10; // r8
  __int64 *v11; // rsi
  unsigned __int64 *v12; // rcx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rax
  bool v17; // al
  unsigned int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // edi
  unsigned __int32 *v21; // rax
  unsigned __int32 v22; // ecx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r13
  __int64 v32; // r12
  __int64 v33; // rcx
  __m128i *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rax
  int v40; // r11d
  unsigned __int32 *v41; // rdx
  int v42; // eax
  int v43; // ecx
  __int64 v44; // rdi
  int v45; // r9d
  int v46; // r9d
  __int64 v47; // r10
  unsigned int v48; // eax
  unsigned __int32 v49; // r8d
  int v50; // edi
  unsigned __int32 *v51; // rsi
  int v52; // edi
  int v53; // edi
  __int64 v54; // r9
  int v55; // esi
  __int64 v56; // rbx
  unsigned __int32 *v57; // rax
  unsigned __int32 v58; // r8d
  _BYTE *v59; // [rsp+0h] [rbp-C0h]
  __int64 v60; // [rsp+8h] [rbp-B8h]
  _QWORD v61[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v63[2]; // [rsp+30h] [rbp-90h] BYREF
  __m128i v64; // [rsp+40h] [rbp-80h] BYREF
  __int16 v65; // [rsp+50h] [rbp-70h]
  _QWORD v66[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v67; // [rsp+80h] [rbp-40h]

  v5 = a3;
  if ( a2[1] )
  {
    v59 = sub_121DBC0((__int64 *)a1, (__int64)a2, a4);
    if ( !v59 )
    {
      v32 = *(_QWORD *)a1;
      sub_8FD6D0((__int64)v61, "unable to create block named '", a2);
      if ( v61[1] == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v34 = (__m128i *)sub_2241490(v61, "'", 1, v33);
      v63[0] = &v64;
      if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
      {
        v64 = _mm_loadu_si128(v34 + 1);
      }
      else
      {
        v63[0] = v34->m128i_i64[0];
        v64.m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v63[1] = v34->m128i_i64[1];
      v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
      v34->m128i_i64[1] = 0;
      v34[1].m128i_i8[0] = 0;
      v67 = 260;
      v66[0] = v63;
      sub_11FD800(v32 + 176, a4, (__int64)v66, 1);
      if ( (__m128i *)v63[0] != &v64 )
        j_j___libc_free_0(v63[0], v64.m128i_i64[0] + 1);
      if ( (__int64 *)v61[0] != &v62 )
        j_j___libc_free_0(v61[0], v62 + 1);
      return v59;
    }
  }
  else
  {
    if ( a3 == -1 )
    {
      v5 = *(_DWORD *)(a1 + 144);
      v8 = v5;
    }
    else
    {
      v8 = a3;
      if ( (unsigned __int8)sub_120EA00(
                              *(_QWORD *)a1,
                              a4,
                              (__int64)"label",
                              5,
                              (__int64)byte_3F871B3,
                              0,
                              *(_DWORD *)(a1 + 144),
                              a3) )
        return 0;
    }
    v59 = sub_121E0D0(a1, v8, a4);
    if ( !v59 )
    {
      v44 = *(_QWORD *)a1;
      v64.m128i_i32[0] = v5;
      v65 = 2563;
      v63[0] = "unable to create block numbered '";
      v66[0] = v63;
      v67 = 770;
      v66[2] = "'";
      sub_11FD800(v44 + 176, a4, (__int64)v66, 1);
      return v59;
    }
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = (unsigned __int64 *)*((_QWORD *)v59 + 4);
  v11 = (__int64 *)(v9 + 72);
  v12 = (unsigned __int64 *)(v59 + 24);
  if ( (unsigned __int64 *)(v9 + 72) != v10 && v12 != (unsigned __int64 *)v11 )
    sub_B2C300(v9, v11, v9, v12, v10);
  if ( !a2[1] )
  {
    v13 = a1 + 72;
    v14 = a1 + 72;
    if ( *(_QWORD *)(a1 + 80) )
    {
      v15 = *(_QWORD *)(a1 + 80);
      while ( 1 )
      {
        while ( v5 > *(_DWORD *)(v15 + 32) )
        {
          v15 = *(_QWORD *)(v15 + 24);
          if ( !v15 )
            goto LABEL_15;
        }
        v16 = *(_QWORD *)(v15 + 16);
        if ( v5 >= *(_DWORD *)(v15 + 32) )
          break;
        v14 = v15;
        v15 = *(_QWORD *)(v15 + 16);
        if ( !v16 )
        {
LABEL_15:
          v17 = v13 == v14;
          goto LABEL_16;
        }
      }
      v35 = *(_QWORD *)(v15 + 24);
      if ( v35 )
      {
        do
        {
          while ( 1 )
          {
            v36 = *(_QWORD *)(v35 + 16);
            v37 = *(_QWORD *)(v35 + 24);
            if ( v5 < *(_DWORD *)(v35 + 32) )
              break;
            v35 = *(_QWORD *)(v35 + 24);
            if ( !v37 )
              goto LABEL_44;
          }
          v14 = v35;
          v35 = *(_QWORD *)(v35 + 16);
        }
        while ( v36 );
      }
LABEL_44:
      while ( v16 )
      {
        while ( 1 )
        {
          v38 = *(_QWORD *)(v16 + 24);
          if ( v5 <= *(_DWORD *)(v16 + 32) )
            break;
          v16 = *(_QWORD *)(v16 + 24);
          if ( !v38 )
            goto LABEL_47;
        }
        v15 = v16;
        v16 = *(_QWORD *)(v16 + 16);
      }
LABEL_47:
      if ( *(_QWORD *)(a1 + 88) != v15 || v13 != v14 )
      {
        for ( ; v15 != v14; --*(_QWORD *)(a1 + 104) )
        {
          v60 = v15;
          v15 = sub_220EF30(v15);
          v39 = sub_220F330(v60, a1 + 72);
          j_j___libc_free_0(v39, 56);
        }
        goto LABEL_19;
      }
    }
    else
    {
      v17 = 1;
LABEL_16:
      if ( *(_QWORD *)(a1 + 88) != v14 || !v17 )
      {
LABEL_19:
        v18 = *(_DWORD *)(a1 + 136);
        if ( v18 )
        {
          v19 = *(_QWORD *)(a1 + 120);
          v20 = (v18 - 1) & (37 * v5);
          v21 = (unsigned __int32 *)(v19 + 16LL * v20);
          v22 = *v21;
          if ( v5 == *v21 )
          {
LABEL_21:
            *(_DWORD *)(a1 + 144) = v5 + 1;
            return v59;
          }
          v40 = 1;
          v41 = 0;
          while ( v22 != -1 )
          {
            if ( v22 != -2 || v41 )
              v21 = v41;
            v20 = (v18 - 1) & (v40 + v20);
            v22 = *(_DWORD *)(v19 + 16LL * v20);
            if ( v5 == v22 )
              goto LABEL_21;
            ++v40;
            v41 = v21;
            v21 = (unsigned __int32 *)(v19 + 16LL * v20);
          }
          if ( !v41 )
            v41 = v21;
          v42 = *(_DWORD *)(a1 + 128);
          ++*(_QWORD *)(a1 + 112);
          v43 = v42 + 1;
          if ( 4 * (v42 + 1) < 3 * v18 )
          {
            if ( v18 - *(_DWORD *)(a1 + 132) - v43 > v18 >> 3 )
            {
LABEL_61:
              *(_DWORD *)(a1 + 128) = v43;
              if ( *v41 != -1 )
                --*(_DWORD *)(a1 + 132);
              *v41 = v5;
              *((_QWORD *)v41 + 1) = v59;
              goto LABEL_21;
            }
            sub_1247200(a1 + 112, v18);
            v52 = *(_DWORD *)(a1 + 136);
            if ( v52 )
            {
              v53 = v52 - 1;
              v54 = *(_QWORD *)(a1 + 120);
              v55 = 1;
              LODWORD(v56) = v53 & (37 * v5);
              v43 = *(_DWORD *)(a1 + 128) + 1;
              v57 = 0;
              v41 = (unsigned __int32 *)(v54 + 16LL * (unsigned int)v56);
              v58 = *v41;
              if ( v5 != *v41 )
              {
                while ( v58 != -1 )
                {
                  if ( v58 == -2 && !v57 )
                    v57 = v41;
                  v56 = v53 & (unsigned int)(v56 + v55);
                  v41 = (unsigned __int32 *)(v54 + 16 * v56);
                  v58 = *v41;
                  if ( v5 == *v41 )
                    goto LABEL_61;
                  ++v55;
                }
                if ( v57 )
                  v41 = v57;
              }
              goto LABEL_61;
            }
LABEL_97:
            ++*(_DWORD *)(a1 + 128);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 112);
        }
        sub_1247200(a1 + 112, 2 * v18);
        v45 = *(_DWORD *)(a1 + 136);
        if ( v45 )
        {
          v46 = v45 - 1;
          v47 = *(_QWORD *)(a1 + 120);
          v48 = v46 & (37 * v5);
          v43 = *(_DWORD *)(a1 + 128) + 1;
          v41 = (unsigned __int32 *)(v47 + 16LL * v48);
          v49 = *v41;
          if ( v5 != *v41 )
          {
            v50 = 1;
            v51 = 0;
            while ( v49 != -1 )
            {
              if ( v49 == -2 && !v51 )
                v51 = v41;
              v48 = v46 & (v50 + v48);
              v41 = (unsigned __int32 *)(v47 + 16LL * v48);
              v49 = *v41;
              if ( v5 == *v41 )
                goto LABEL_61;
              ++v50;
            }
            if ( v51 )
              v41 = v51;
          }
          goto LABEL_61;
        }
        goto LABEL_97;
      }
    }
    sub_1206180(*(_QWORD *)(a1 + 80));
    *(_QWORD *)(a1 + 88) = v13;
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 96) = v13;
    *(_QWORD *)(a1 + 104) = 0;
    goto LABEL_19;
  }
  v24 = sub_1216F80(a1 + 16, (__int64)a2);
  v26 = v25;
  v27 = v24;
  if ( v24 == *(_QWORD *)(a1 + 40) && v25 == a1 + 24 )
  {
    sub_1207330(*(_QWORD **)(a1 + 32));
    *(_QWORD *)(a1 + 40) = v26;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 48) = v26;
    *(_QWORD *)(a1 + 56) = 0;
  }
  else if ( v25 != v24 )
  {
    do
    {
      v28 = v27;
      v27 = sub_220EF30(v27);
      v29 = sub_220F330(v28, a1 + 24);
      v30 = *(_QWORD *)(v29 + 32);
      v31 = v29;
      if ( v30 != v29 + 48 )
        j_j___libc_free_0(v30, *(_QWORD *)(v29 + 48) + 1LL);
      j_j___libc_free_0(v31, 80);
      --*(_QWORD *)(a1 + 56);
    }
    while ( v26 != v27 );
  }
  return v59;
}

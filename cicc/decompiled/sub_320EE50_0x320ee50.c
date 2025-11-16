// Function: sub_320EE50
// Address: 0x320ee50
//
__int64 __fastcall sub_320EE50(unsigned __int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  int v4; // eax
  __int64 v5; // r13
  unsigned int v6; // eax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int8 *v13; // rdx
  _BYTE *v14; // r13
  unsigned __int64 *v15; // r14
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rsi
  bool v19; // zf
  const __m128i *v20; // r14
  const __m128i *v21; // r12
  signed __int64 v22; // r13
  __m128i *v23; // rax
  __m128i *v24; // rcx
  __m128i *v25; // rdx
  __m128i *v26; // r8
  _QWORD *v27; // rax
  unsigned __int64 v28; // rdi
  __int64 result; // rax
  bool v30; // cc
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // r15
  unsigned __int64 v33; // r14
  unsigned __int64 v34; // r12
  unsigned __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 v37; // rsi
  unsigned int v38; // edi
  unsigned int v39; // edx
  __int64 v40; // r8
  __int64 v41; // r8
  __int64 v42; // rdx
  _QWORD *v43; // r12
  unsigned int v44; // r9d
  __int64 *v45; // rcx
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r14
  _QWORD *v50; // r13
  unsigned __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int64 v54; // r13
  _QWORD *v55; // rcx
  _QWORD *v56; // rdx
  unsigned __int64 v57; // rdi
  __int64 v58; // rdx
  int v59; // r13d
  unsigned int v60; // eax
  _QWORD *v61; // rdi
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rcx
  _QWORD *i; // rdx
  unsigned __int64 v67; // rdi
  _QWORD *v68; // rax
  int v69; // ecx
  int v70; // r11d
  unsigned int v71; // r9d
  __int64 v72; // [rsp+0h] [rbp-80h]
  __int64 v74; // [rsp+10h] [rbp-70h]
  __int64 v75; // [rsp+18h] [rbp-68h]
  __int64 v76; // [rsp+20h] [rbp-60h]
  __int64 *j; // [rsp+20h] [rbp-60h]
  int v78; // [rsp+28h] [rbp-58h]
  __int64 v79; // [rsp+28h] [rbp-58h]
  __int64 *v80; // [rsp+28h] [rbp-58h]
  _BYTE *v81; // [rsp+30h] [rbp-50h] BYREF
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 (__fastcall *v83)(const __m128i **, const __m128i *, int); // [rsp+40h] [rbp-40h]
  void (__fastcall *v84)(_QWORD *, __int64, __int64, __int64 *); // [rsp+48h] [rbp-38h]

  v75 = *a2;
  sub_B92180(*a2);
  sub_320D7D0(a1);
  v3 = *(_QWORD *)(a1 + 296);
  if ( v3 )
    sub_320E0C0(a1, v3, *(_QWORD *)(a1 + 792) + 344LL, *(_QWORD *)(a1 + 792) + 152LL, *(_QWORD *)(a1 + 792) + 256LL);
  v4 = *(_DWORD *)(a1 + 856);
  ++*(_QWORD *)(a1 + 840);
  v78 = v4;
  if ( v4 || *(_DWORD *)(a1 + 860) )
  {
    v5 = *(_QWORD *)(a1 + 848);
    v6 = 4 * v4;
    v72 = 112LL * *(unsigned int *)(a1 + 864);
    v76 = v5 + v72;
    if ( (unsigned int)(4 * v78) < 0x40 )
      v6 = 64;
    if ( *(_DWORD *)(a1 + 864) <= v6 )
    {
      if ( v5 != v5 + 112LL * *(unsigned int *)(a1 + 864) )
      {
        do
        {
          if ( *(_QWORD *)v5 != -4096 )
          {
            if ( *(_QWORD *)v5 != -8192 )
            {
              v79 = *(_QWORD *)(v5 + 8);
              v7 = v79 + 88LL * *(unsigned int *)(v5 + 16);
              if ( v79 != v7 )
              {
                do
                {
                  v7 -= 88LL;
                  if ( *(_BYTE *)(v7 + 80) )
                  {
                    v30 = *(_DWORD *)(v7 + 72) <= 0x40u;
                    *(_BYTE *)(v7 + 80) = 0;
                    if ( !v30 )
                    {
                      v31 = *(_QWORD *)(v7 + 64);
                      if ( v31 )
                        j_j___libc_free_0_0(v31);
                    }
                  }
                  v8 = *(_QWORD *)(v7 + 40);
                  v9 = v8 + 40LL * *(unsigned int *)(v7 + 48);
                  if ( v8 != v9 )
                  {
                    do
                    {
                      v9 -= 40LL;
                      v10 = *(_QWORD *)(v9 + 8);
                      if ( v10 != v9 + 24 )
                        _libc_free(v10);
                    }
                    while ( v8 != v9 );
                    v8 = *(_QWORD *)(v7 + 40);
                  }
                  if ( v8 != v7 + 56 )
                    _libc_free(v8);
                  sub_C7D6A0(*(_QWORD *)(v7 + 16), 12LL * *(unsigned int *)(v7 + 32), 4);
                }
                while ( v79 != v7 );
                v7 = *(_QWORD *)(v5 + 8);
              }
              if ( v7 != v5 + 24 )
                _libc_free(v7);
            }
            *(_QWORD *)v5 = -4096;
          }
          v5 += 112;
        }
        while ( v5 != v76 );
      }
      goto LABEL_26;
    }
    while ( 1 )
    {
      while ( *(_QWORD *)v5 == -8192 )
      {
LABEL_80:
        v5 += 112;
        if ( v5 == v76 )
          goto LABEL_116;
      }
      if ( *(_QWORD *)v5 != -4096 )
      {
        v74 = *(_QWORD *)(v5 + 8);
        v32 = v74 + 88LL * *(unsigned int *)(v5 + 16);
        if ( v74 != v32 )
        {
          do
          {
            v32 -= 88LL;
            if ( *(_BYTE *)(v32 + 80) )
            {
              v30 = *(_DWORD *)(v32 + 72) <= 0x40u;
              *(_BYTE *)(v32 + 80) = 0;
              if ( !v30 )
              {
                v67 = *(_QWORD *)(v32 + 64);
                if ( v67 )
                  j_j___libc_free_0_0(v67);
              }
            }
            v33 = *(_QWORD *)(v32 + 40);
            v34 = v33 + 40LL * *(unsigned int *)(v32 + 48);
            if ( v33 != v34 )
            {
              do
              {
                v34 -= 40LL;
                v35 = *(_QWORD *)(v34 + 8);
                if ( v35 != v34 + 24 )
                  _libc_free(v35);
              }
              while ( v33 != v34 );
              v33 = *(_QWORD *)(v32 + 40);
            }
            if ( v33 != v32 + 56 )
              _libc_free(v33);
            sub_C7D6A0(*(_QWORD *)(v32 + 16), 12LL * *(unsigned int *)(v32 + 32), 4);
          }
          while ( v74 != v32 );
          v32 = *(_QWORD *)(v5 + 8);
        }
        if ( v32 != v5 + 24 )
          _libc_free(v32);
        goto LABEL_80;
      }
      v5 += 112;
      if ( v5 == v76 )
      {
LABEL_116:
        v58 = *(unsigned int *)(a1 + 864);
        if ( v78 )
        {
          v59 = 64;
          if ( v78 != 1 )
          {
            _BitScanReverse(&v60, v78 - 1);
            v59 = 1 << (33 - (v60 ^ 0x1F));
            if ( v59 < 64 )
              v59 = 64;
          }
          v61 = *(_QWORD **)(a1 + 848);
          if ( (_DWORD)v58 == v59 )
          {
            *(_QWORD *)(a1 + 856) = 0;
            v68 = &v61[14 * v58];
            do
            {
              if ( v61 )
                *v61 = -4096;
              v61 += 14;
            }
            while ( v68 != v61 );
          }
          else
          {
            sub_C7D6A0((__int64)v61, v72, 8);
            v62 = ((((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                     | (4 * v59 / 3u + 1)
                     | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                   | (4 * v59 / 3u + 1)
                   | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                   | (4 * v59 / 3u + 1)
                   | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                 | (4 * v59 / 3u + 1)
                 | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 16;
            v63 = (v62
                 | (((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                     | (4 * v59 / 3u + 1)
                     | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                   | (4 * v59 / 3u + 1)
                   | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                   | (4 * v59 / 3u + 1)
                   | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                 | (4 * v59 / 3u + 1)
                 | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 864) = v63;
            v64 = (_QWORD *)sub_C7D670(112 * v63, 8);
            v65 = *(unsigned int *)(a1 + 864);
            *(_QWORD *)(a1 + 856) = 0;
            *(_QWORD *)(a1 + 848) = v64;
            for ( i = &v64[14 * v65]; i != v64; v64 += 14 )
            {
              if ( v64 )
                *v64 = -4096;
            }
          }
          break;
        }
        if ( (_DWORD)v58 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 848), v72, 8);
          *(_QWORD *)(a1 + 848) = 0;
          *(_QWORD *)(a1 + 856) = 0;
          *(_DWORD *)(a1 + 864) = 0;
          break;
        }
LABEL_26:
        *(_QWORD *)(a1 + 856) = 0;
        break;
      }
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 792) + 489LL) || (result = sub_B92180(v75), (*(_BYTE *)(result + 35) & 2) != 0) )
  {
    for ( j = (__int64 *)a2[41]; a2 + 40 != j; j = (__int64 *)j[1] )
    {
      v11 = j[7];
      v80 = j + 6;
      if ( (__int64 *)v11 != j + 6 )
      {
        do
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(v11 + 48);
            v13 = (unsigned __int8 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v12 & 7) == 3 )
            {
              if ( v13[6] )
              {
                v14 = *(_BYTE **)&v13[8 * *(int *)v13 + 16 + 8 * (__int64)(v13[5] + v13[4])];
                if ( v14 )
                {
                  v15 = *(unsigned __int64 **)(a1 + 792);
                  if ( *v14 > 0x24u )
                  {
                    v14 = 0;
                  }
                  else if ( ((1LL << *v14) & 0x140000F000LL) == 0 )
                  {
                    v14 = 0;
                  }
                  v16 = sub_3211FB0(a1, v11);
                  v17 = sub_3211F40(a1, v11);
                  v81 = v14;
                  v82 = v16;
                  v83 = (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))v17;
                  v18 = v15[50];
                  if ( v18 == v15[51] )
                  {
                    sub_31FCBC0(v15 + 49, (char *)v18, &v81);
                  }
                  else
                  {
                    if ( v18 )
                    {
                      *(_QWORD *)v18 = v14;
                      *(_QWORD *)(v18 + 8) = v82;
                      *(_QWORD *)(v18 + 16) = v83;
                      v18 = v15[50];
                    }
                    v15[50] = v18 + 24;
                  }
                }
              }
            }
            if ( (*(_BYTE *)v11 & 4) == 0 )
              break;
            v11 = *(_QWORD *)(v11 + 8);
            if ( v80 == (__int64 *)v11 )
              goto LABEL_45;
          }
          while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
            v11 = *(_QWORD *)(v11 + 8);
          v11 = *(_QWORD *)(v11 + 8);
        }
        while ( v80 != (__int64 *)v11 );
      }
LABEL_45:
      ;
    }
    v19 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 264LL) == 36;
    v81 = (_BYTE *)a1;
    v84 = sub_31FCF40;
    v83 = sub_31F3D60;
    v82 = (__int64)a2;
    sub_31F99A0((__int64)a2, v19, (__int64)&v81);
    if ( v83 )
      v83((const __m128i **)&v81, (const __m128i *)&v81, 3);
    v20 = (const __m128i *)a2[70];
    v21 = (const __m128i *)a2[69];
    v22 = (char *)v20 - (char *)v21;
    if ( (unsigned __int64)((char *)v20 - (char *)v21) > 0x7FFFFFFFFFFFFFF0LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    if ( v22 )
    {
      v23 = (__m128i *)sub_22077B0(a2[70] - (_QWORD)v21);
      v24 = v23;
      v25 = (__m128i *)((char *)v23 + v22);
      if ( v21 == v20 )
      {
        v26 = v23;
      }
      else
      {
        v26 = (__m128i *)((char *)v23 + v22);
        do
        {
          if ( v23 )
            *v23 = _mm_loadu_si128(v21);
          ++v23;
          ++v21;
        }
        while ( v23 != v25 );
      }
    }
    else
    {
      v25 = 0;
      v24 = 0;
      v26 = 0;
    }
    v27 = *(_QWORD **)(a1 + 792);
    v28 = v27[46];
    v27[46] = v24;
    v27[47] = v26;
    v27[48] = v25;
    if ( v28 )
      j_j___libc_free_0(v28);
    result = *(_QWORD *)(a1 + 792);
    *(_QWORD *)(result + 448) = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 400LL);
  }
  else
  {
    v36 = *(_DWORD *)(a1 + 1080);
    v37 = *(_QWORD *)(a1 + 1064);
    if ( v36 )
    {
      v38 = v36 - 1;
      v39 = (v36 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
      result = v37 + 16LL * v39;
      v40 = *(_QWORD *)result;
      if ( v75 == *(_QWORD *)result )
      {
LABEL_89:
        if ( result != v37 + 16LL * v36 )
        {
          v41 = *(_QWORD *)(a1 + 1088);
          v42 = *(unsigned int *)(a1 + 1096);
          v43 = (_QWORD *)(v41 + 16LL * *(unsigned int *)(result + 8));
          result = v41 + 16 * v42;
          if ( (_QWORD *)result != v43 )
          {
            v44 = v38 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
            v45 = (__int64 *)(v37 + 16LL * v44);
            v46 = *v45;
            if ( *v43 == *v45 )
            {
LABEL_92:
              *v45 = -8192;
              v47 = *(unsigned int *)(a1 + 1096);
              --*(_DWORD *)(a1 + 1072);
              v41 = *(_QWORD *)(a1 + 1088);
              ++*(_DWORD *)(a1 + 1076);
              LODWORD(v42) = v47;
              result = v41 + 16 * v47;
            }
            else
            {
              v69 = 1;
              while ( v46 != -4096 )
              {
                v70 = v69 + 1;
                v44 = v38 & (v69 + v44);
                v45 = (__int64 *)(v37 + 16LL * v44);
                v46 = *v45;
                if ( *v43 == *v45 )
                  goto LABEL_92;
                v69 = v70;
              }
            }
            v48 = result - (_QWORD)(v43 + 2);
            v49 = v48 >> 4;
            if ( v48 > 0 )
            {
              v50 = v43;
              do
              {
                v51 = v50[1];
                *v50 = v50[2];
                v52 = v50[3];
                v50[3] = 0;
                v50[1] = v52;
                if ( v51 )
                {
                  sub_31FB410(v51);
                  j_j___libc_free_0(v51);
                }
                v50 += 2;
                --v49;
              }
              while ( v49 );
              LODWORD(v42) = *(_DWORD *)(a1 + 1096);
              v41 = *(_QWORD *)(a1 + 1088);
            }
            v53 = (unsigned int)(v42 - 1);
            *(_DWORD *)(a1 + 1096) = v53;
            v54 = *(_QWORD *)(v41 + 16 * v53 + 8);
            if ( v54 )
            {
              sub_31FB410(*(_QWORD *)(v41 + 16 * v53 + 8));
              j_j___libc_free_0(v54);
              v41 = *(_QWORD *)(a1 + 1088);
            }
            result = v41 + 16LL * *(unsigned int *)(a1 + 1096);
            if ( v43 != (_QWORD *)result )
            {
              result = *(unsigned int *)(a1 + 1072);
              if ( (_DWORD)result )
              {
                v55 = *(_QWORD **)(a1 + 1064);
                v56 = &v55[2 * *(unsigned int *)(a1 + 1080)];
                if ( v55 != v56 )
                {
                  while ( 1 )
                  {
                    result = (__int64)v55;
                    if ( *v55 != -4096 && *v55 != -8192 )
                      break;
                    v55 += 2;
                    if ( v56 == v55 )
                      goto LABEL_58;
                  }
                  while ( (_QWORD *)result != v56 )
                  {
                    v57 = *(unsigned int *)(result + 8);
                    if ( ((__int64)v43 - v41) >> 4 < v57 )
                      *(_DWORD *)(result + 8) = v57 - 1;
                    result += 16;
                    if ( (_QWORD *)result == v56 )
                      break;
                    while ( *(_QWORD *)result == -4096 || *(_QWORD *)result == -8192 )
                    {
                      result += 16;
                      if ( v56 == (_QWORD *)result )
                        goto LABEL_58;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        result = 1;
        while ( v40 != -4096 )
        {
          v71 = result + 1;
          v39 = v38 & (result + v39);
          result = v37 + 16LL * v39;
          v40 = *(_QWORD *)result;
          if ( v75 == *(_QWORD *)result )
            goto LABEL_89;
          result = v71;
        }
      }
    }
  }
LABEL_58:
  *(_QWORD *)(a1 + 792) = 0;
  return result;
}

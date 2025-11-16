// Function: sub_1037A30
// Address: 0x1037a30
//
__int64 __fastcall sub_1037A30(__int64 a1, unsigned __int8 *a2, char a3)
{
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  unsigned int v10; // esi
  __int64 v11; // rcx
  int v12; // r10d
  unsigned __int8 **v13; // r14
  unsigned int v14; // edx
  _QWORD *v15; // rax
  unsigned __int8 *v16; // r8
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // r8
  unsigned int v21; // ecx
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  int v24; // r14d
  _QWORD *v25; // rax
  int v26; // eax
  int v27; // ecx
  unsigned __int8 *v28; // r15
  __int64 v29; // r9
  char v30; // al
  __int64 *v31; // rdx
  unsigned __int8 v32; // al
  char v33; // al
  __int64 v34; // rdx
  _QWORD *v35; // rax
  unsigned __int8 **v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdi
  unsigned __int8 **v41; // rax
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  int v44; // eax
  int v45; // edx
  __int64 v46; // rdi
  unsigned int v47; // eax
  unsigned __int8 *v48; // rsi
  int v49; // r9d
  unsigned __int8 **v50; // r8
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rdx
  _QWORD *j; // rdx
  int v56; // eax
  int v57; // eax
  __int64 v58; // rsi
  int v59; // r8d
  unsigned int v60; // r15d
  unsigned __int8 **v61; // rdi
  unsigned __int8 *v62; // rdx
  __int64 v63; // rdx
  unsigned int v64; // eax
  __int64 v65; // rdx
  __int64 v66; // [rsp+0h] [rbp-70h]
  __int64 v67; // [rsp+8h] [rbp-68h]
  __m128i v68; // [rsp+10h] [rbp-60h] BYREF
  __int64 v69; // [rsp+20h] [rbp-50h]
  __int64 v70; // [rsp+28h] [rbp-48h]
  __int64 v71; // [rsp+30h] [rbp-40h]
  __int64 v72; // [rsp+38h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 1000);
  ++*(_QWORD *)(a1 + 984);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 1004) )
      goto LABEL_7;
    v7 = *(unsigned int *)(a1 + 1008);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 992), 16LL * (unsigned int)v7, 8);
      *(_QWORD *)(a1 + 992) = 0;
      *(_QWORD *)(a1 + 1000) = 0;
      *(_DWORD *)(a1 + 1008) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v21 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 1008);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v21 = 64;
  if ( (unsigned int)v7 <= v21 )
  {
LABEL_4:
    v8 = *(_QWORD **)(a1 + 992);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 1000) = 0;
    goto LABEL_7;
  }
  v22 = v6 - 1;
  if ( v22 )
  {
    _BitScanReverse(&v22, v22);
    v23 = *(_QWORD **)(a1 + 992);
    v24 = 1 << (33 - (v22 ^ 0x1F));
    if ( v24 < 64 )
      v24 = 64;
    if ( (_DWORD)v7 == v24 )
    {
      *(_QWORD *)(a1 + 1000) = 0;
      v25 = &v23[2 * (unsigned int)v7];
      do
      {
        if ( v23 )
          *v23 = -4096;
        v23 += 2;
      }
      while ( v25 != v23 );
      goto LABEL_7;
    }
  }
  else
  {
    v23 = *(_QWORD **)(a1 + 992);
    v24 = 64;
  }
  sub_C7D6A0((__int64)v23, 16LL * (unsigned int)v7, 8);
  v51 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
       | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
       | (4 * v24 / 3u + 1)
       | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 16;
  v52 = (v51
       | (((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
       | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
       | (4 * v24 / 3u + 1)
       | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 1008) = v52;
  v53 = (_QWORD *)sub_C7D670(16 * v52, 8);
  v54 = *(unsigned int *)(a1 + 1008);
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 992) = v53;
  for ( j = &v53[2 * v54]; j != v53; v53 += 2 )
  {
    if ( v53 )
      *v53 = -4096;
  }
LABEL_7:
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_61;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (_QWORD *)(v11 + 16LL * v14);
  v16 = (unsigned __int8 *)*v15;
  if ( (unsigned __int8 *)*v15 != a2 )
  {
    while ( v16 != (unsigned __int8 *)-4096LL )
    {
      if ( !v13 && v16 == (unsigned __int8 *)-8192LL )
        v13 = (unsigned __int8 **)v15;
      v14 = (v10 - 1) & (v12 + v14);
      v15 = (_QWORD *)(v11 + 16LL * v14);
      v16 = (unsigned __int8 *)*v15;
      if ( (unsigned __int8 *)*v15 == a2 )
        goto LABEL_9;
      ++v12;
    }
    if ( !v13 )
      v13 = (unsigned __int8 **)v15;
    v26 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v27 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 20) - v27 > v10 >> 3 )
      {
LABEL_34:
        *(_DWORD *)(a1 + 16) = v27;
        if ( *v13 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a1 + 20);
        *v13 = a2;
        v28 = a2;
        v17 = (__int64 *)(v13 + 1);
        *v17 = 0;
LABEL_37:
        v29 = *((_QWORD *)a2 + 5);
        if ( *a2 == 61 )
        {
          v67 = *((_QWORD *)a2 + 5);
          v30 = sub_102D1C0(a1 + 872, (__int64)a2);
          v29 = v67;
          if ( v30 )
          {
            v19 = 0x2000000000000003LL;
            *v17 = 0x2000000000000003LL;
            return v19;
          }
        }
        if ( *(unsigned __int8 **)(v29 + 56) == a2 + 24 )
        {
          v43 = *(_QWORD *)(*(_QWORD *)(v29 + 72) + 80LL);
          if ( v43 && v29 == v43 - 24 )
            *v17 = 0x4000000000000003LL;
          else
            *v17 = 0x2000000000000003LL;
        }
        else
        {
          v31 = *(__int64 **)(a1 + 272);
          v66 = v29;
          v68.m128i_i64[0] = 0;
          v68.m128i_i64[1] = -1;
          v69 = 0;
          v70 = 0;
          v71 = 0;
          v72 = 0;
          v32 = sub_102A4D0(a2, &v68, v31);
          if ( v68.m128i_i64[0] )
          {
            v33 = ((v32 >> 1) ^ 1) & 1;
            if ( *a2 == 85 )
            {
              v65 = *((_QWORD *)a2 - 4);
              if ( v65 )
              {
                if ( !*(_BYTE *)v65
                  && *(_QWORD *)(v65 + 24) == *((_QWORD *)a2 + 10)
                  && (*(_BYTE *)(v65 + 33) & 0x20) != 0 )
                {
                  v33 |= *(_DWORD *)(v65 + 36) == 211;
                }
              }
            }
            *v17 = sub_1037870(a1, (__int64)&v68, v33, (_QWORD *)v28 + 3, 0, v66, a2, 0, a3);
          }
          else if ( (unsigned __int8)(*a2 - 34) <= 0x33u
                 && (v63 = 0x8000000000041LL, _bittest64(&v63, (unsigned int)*a2 - 34)) )
          {
            v64 = sub_CF5CA0(*(_QWORD *)(a1 + 256), (__int64)a2);
            *v17 = sub_102AD20(
                     a1,
                     a2,
                     (((unsigned __int8)((v64 >> 6) | (v64 >> 4) | v64 | (v64 >> 2)) >> 1) ^ 1) & 1,
                     (_QWORD *)v28 + 3,
                     0,
                     v66);
          }
          else
          {
            *v17 = 0x6000000000000003LL;
          }
          v34 = *v17 & 7;
          if ( (unsigned int)v34 > 2 )
          {
            if ( (_DWORD)v34 != 3 )
              BUG();
          }
          else
          {
            v68.m128i_i64[0] = *v17 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v68.m128i_i64[0] )
            {
              v35 = sub_1030100(a1 + 192, v68.m128i_i64);
              v40 = (__int64)v35;
              if ( !*((_BYTE *)v35 + 28) )
                goto LABEL_59;
              v41 = (unsigned __int8 **)v35[1];
              v37 = *(unsigned int *)(v40 + 20);
              v36 = &v41[v37];
              if ( v41 != v36 )
              {
                while ( *v41 != a2 )
                {
                  if ( v36 == ++v41 )
                    goto LABEL_80;
                }
                return *v17;
              }
LABEL_80:
              if ( (unsigned int)v37 < *(_DWORD *)(v40 + 16) )
              {
                *(_DWORD *)(v40 + 20) = v37 + 1;
                *v36 = a2;
                ++*(_QWORD *)v40;
              }
              else
              {
LABEL_59:
                sub_C8CC70(v40, (__int64)a2, (__int64)v36, v37, v38, v39);
              }
            }
          }
        }
        return *v17;
      }
      sub_102FCF0(a1, v10);
      v56 = *(_DWORD *)(a1 + 24);
      if ( v56 )
      {
        v57 = v56 - 1;
        v58 = *(_QWORD *)(a1 + 8);
        v59 = 1;
        v60 = v57 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v61 = 0;
        v27 = *(_DWORD *)(a1 + 16) + 1;
        v13 = (unsigned __int8 **)(v58 + 16LL * v60);
        v62 = *v13;
        if ( *v13 != a2 )
        {
          while ( v62 != (unsigned __int8 *)-4096LL )
          {
            if ( !v61 && v62 == (unsigned __int8 *)-8192LL )
              v61 = v13;
            v60 = v57 & (v59 + v60);
            v13 = (unsigned __int8 **)(v58 + 16LL * v60);
            v62 = *v13;
            if ( *v13 == a2 )
              goto LABEL_34;
            ++v59;
          }
          if ( v61 )
            v13 = v61;
        }
        goto LABEL_34;
      }
LABEL_103:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_61:
    sub_102FCF0(a1, 2 * v10);
    v44 = *(_DWORD *)(a1 + 24);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 8);
      v47 = (v44 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(a1 + 16) + 1;
      v13 = (unsigned __int8 **)(v46 + 16LL * v47);
      v48 = *v13;
      if ( *v13 != a2 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != (unsigned __int8 *)-4096LL )
        {
          if ( !v50 && v48 == (unsigned __int8 *)-8192LL )
            v50 = v13;
          v47 = v45 & (v49 + v47);
          v13 = (unsigned __int8 **)(v46 + 16LL * v47);
          v48 = *v13;
          if ( *v13 == a2 )
            goto LABEL_34;
          ++v49;
        }
        if ( v50 )
          v13 = v50;
      }
      goto LABEL_34;
    }
    goto LABEL_103;
  }
LABEL_9:
  v17 = v15 + 1;
  v18 = v15[1];
  v19 = v18;
  if ( (v18 & 7) == 0 )
  {
    v42 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v42 )
    {
      v28 = (unsigned __int8 *)v42;
      sub_1029DA0(a1 + 192, v42, (__int64)a2);
    }
    else
    {
      v28 = a2;
    }
    goto LABEL_37;
  }
  return v19;
}

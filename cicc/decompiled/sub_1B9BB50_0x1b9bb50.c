// Function: sub_1B9BB50
// Address: 0x1b9bb50
//
__int64 __fastcall sub_1B9BB50(__int64 a1, __int64 **a2, int a3, unsigned int a4, __m128i a5, __m128i a6, double a7)
{
  __int64 v9; // rdi
  int v10; // eax
  int v11; // edx
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 **v14; // rcx
  unsigned int v15; // ebx
  __int64 *v16; // r13
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned int *v19; // r8
  int v20; // r9d
  __int64 v21; // r15
  unsigned int v22; // edx
  int v23; // r8d
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 **v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rbx
  unsigned int v33; // r13d
  char *v34; // r9
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // r15
  bool v40; // zf
  int v41; // eax
  __int64 *v42; // r14
  int v43; // ebx
  __int64 v44; // rdx
  _QWORD *v45; // rax
  int v46; // r8d
  unsigned __int64 *v47; // r9
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdi
  int v54; // edx
  int v55; // r10d
  __int64 *v56; // [rsp+10h] [rbp-110h]
  unsigned __int64 v57; // [rsp+18h] [rbp-108h]
  __int64 v58; // [rsp+28h] [rbp-F8h]
  int v59; // [rsp+3Ch] [rbp-E4h]
  int v60; // [rsp+40h] [rbp-E0h]
  int v61; // [rsp+48h] [rbp-D8h]
  __int64 v64; // [rsp+58h] [rbp-C8h]
  __int64 v65; // [rsp+58h] [rbp-C8h]
  __int64 v66; // [rsp+58h] [rbp-C8h]
  __int64 v67; // [rsp+58h] [rbp-C8h]
  __int64 v68; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v69; // [rsp+70h] [rbp-B0h]
  __int64 v70[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v71; // [rsp+90h] [rbp-90h]
  __int64 v72; // [rsp+A0h] [rbp-80h] BYREF
  __int128 v73; // [rsp+A8h] [rbp-78h]
  __int128 v74; // [rsp+B8h] [rbp-68h]
  __int64 v75; // [rsp+C8h] [rbp-58h]
  char *v76; // [rsp+D0h] [rbp-50h] BYREF
  __int128 v77; // [rsp+D8h] [rbp-48h] BYREF
  __int64 v78; // [rsp+E8h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 448);
  v10 = *(_DWORD *)(v9 + 96);
  if ( v10 )
  {
    v11 = v10 - 1;
    v12 = *(_QWORD *)(v9 + 80);
    result = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = *(__int64 ***)(v12 + 176 * result);
    if ( a2 == v14 )
      goto LABEL_3;
    v23 = 1;
    while ( v14 != (__int64 **)-8LL )
    {
      result = v11 & (unsigned int)(v23 + result);
      v14 = *(__int64 ***)(v12 + 176LL * (unsigned int)result);
      if ( a2 == v14 )
        goto LABEL_3;
      ++v23;
    }
  }
  result = sub_1BF28F0(v9, a2);
  if ( (_BYTE)result )
  {
LABEL_3:
    if ( a3 )
    {
      v15 = 0;
      do
      {
        v16 = *a2;
        if ( a4 != 1 )
          v16 = sub_16463B0(*a2, a4);
        v17 = sub_157EE30(*(_QWORD *)(a1 + 200));
        WORD4(v73) = 259;
        if ( v17 )
          v17 -= 24;
        v72 = (__int64)"vec.phi";
        v64 = v17;
        v18 = sub_1648B60(64);
        v21 = v18;
        if ( v18 )
        {
          sub_15F1EA0(v18, (__int64)v16, 53, 0, 0, v64);
          *(_DWORD *)(v21 + 56) = 2;
          sub_164B780(v21, &v72);
          sub_1648880(v21, *(_DWORD *)(v21 + 56), 1);
        }
        v22 = v15++;
        result = sub_1B99BD0((unsigned int *)(a1 + 280), (unsigned __int64)a2, v22, v21, v19, v20);
      }
      while ( a3 != v15 );
    }
    return result;
  }
  sub_1B91520(a1, (__int64 *)(a1 + 96), (__int64)a2);
  v24 = *(_QWORD *)(a1 + 448);
  v25 = *(unsigned int *)(v24 + 128);
  if ( (_DWORD)v25 )
  {
    v26 = *(_QWORD *)(v24 + 112);
    v27 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v28 = v26 + 16LL * v27;
    v29 = *(__int64 ***)v28;
    if ( a2 == *(__int64 ***)v28 )
    {
LABEL_18:
      if ( v28 != v26 + 16 * v25 )
      {
        v30 = 11LL * *(unsigned int *)(v28 + 8);
        v31 = *(_QWORD *)(v24 + 136);
        v72 = 6;
        *(_QWORD *)&v73 = 0;
        v32 = v31 + 8 * v30;
        *((_QWORD *)&v73 + 1) = *(_QWORD *)(v32 + 24);
        if ( *((_QWORD *)&v73 + 1) != -8 && *((_QWORD *)&v73 + 1) != 0 && *((_QWORD *)&v73 + 1) != -16 )
          sub_1649AC0((unsigned __int64 *)&v72, *(_QWORD *)(v32 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        LODWORD(v74) = *(_DWORD *)(v32 + 32);
        *((_QWORD *)&v74 + 1) = *(_QWORD *)(v32 + 40);
        v75 = *(_QWORD *)(v32 + 48);
        v76 = (char *)&v77 + 8;
        *(_QWORD *)&v77 = 0x200000000LL;
        v33 = *(_DWORD *)(v32 + 64);
        if ( v33 && &v76 != (char **)(v32 + 56) )
        {
          v34 = (char *)&v77 + 8;
          v35 = 8LL * v33;
          if ( v33 <= 2
            || (sub_16CD150((__int64)&v76, (char *)&v77 + 8, v33, 8, v35, (int)&v77 + 8),
                v34 = v76,
                (v35 = 8LL * *(unsigned int *)(v32 + 64)) != 0) )
          {
            memcpy(v34, *(const void **)(v32 + 56), v35);
          }
          LODWORD(v77) = v33;
        }
        goto LABEL_28;
      }
    }
    else
    {
      v54 = 1;
      while ( v29 != (__int64 **)-8LL )
      {
        v55 = v54 + 1;
        v27 = (v25 - 1) & (v54 + v27);
        v28 = v26 + 16LL * v27;
        v29 = *(__int64 ***)v28;
        if ( a2 == *(__int64 ***)v28 )
          goto LABEL_18;
        v54 = v55;
      }
    }
  }
  a5 = 0;
  v78 = 0;
  v75 = 0;
  v77 = 0;
  v72 = 6;
  v76 = (char *)&v77 + 8;
  DWORD1(v77) = 2;
  v73 = 0;
  v74 = 0;
LABEL_28:
  v36 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
  v58 = sub_1632FA0(v36);
  if ( (_DWORD)v74 == 2 )
  {
    v37 = *(_QWORD *)(a1 + 264);
    v71 = 257;
    v38 = sub_1456040(*((__int64 *)&v74 + 1));
    v39 = sub_1904CF0((__int64 *)(a1 + 96), v37, v38, v70);
    v40 = sub_1B960F0(*(_QWORD *)(a1 + 456), (__int64)a2, a4) == 0;
    v41 = 1;
    if ( v40 )
      v41 = a4;
    v60 = v41;
    if ( a3 )
    {
      v57 = (unsigned __int64)a2;
      v42 = (__int64 *)(a1 + 96);
      v59 = 0;
      v61 = 0;
      do
      {
        v43 = 0;
        if ( v60 )
        {
          do
          {
            v48 = sub_15A0680(*(_QWORD *)v39, (unsigned int)(v43 + v59), 0);
            v69 = 257;
            if ( *(_BYTE *)(v39 + 16) > 0x10u || *(_BYTE *)(v48 + 16) > 0x10u )
            {
              v71 = 257;
              v49 = sub_15FB440(11, (__int64 *)v39, v48, (__int64)v70, 0);
              v50 = *(_QWORD *)(a1 + 104);
              v51 = v49;
              if ( v50 )
              {
                v66 = v49;
                v56 = *(__int64 **)(a1 + 112);
                sub_157E9D0(v50 + 40, v49);
                v51 = v66;
                v52 = *(_QWORD *)(v66 + 24);
                v53 = *v56;
                *(_QWORD *)(v66 + 32) = v56;
                v53 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v66 + 24) = v53 | v52 & 7;
                *(_QWORD *)(v53 + 8) = v66 + 24;
                *v56 = *v56 & 7 | (v66 + 24);
              }
              v67 = v51;
              sub_164B780(v51, &v68);
              sub_12A86E0(v42, v67);
              v44 = v67;
            }
            else
            {
              v44 = sub_15A2B30((__int64 *)v39, v48, 0, 0, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
            }
            v45 = sub_1B19340(
                    (__int64)&v72,
                    (__int64)v42,
                    v44,
                    *(_QWORD **)(*(_QWORD *)(a1 + 16) + 112LL),
                    v58,
                    a5,
                    a6,
                    a7);
            v71 = 259;
            v70[0] = (__int64)"next.gep";
            v65 = (__int64)v45;
            sub_164B780((__int64)v45, v70);
            HIDWORD(v70[0]) = v43++;
            LODWORD(v70[0]) = v61;
            sub_1B9A1B0((unsigned int *)(a1 + 280), v57, (unsigned int *)v70, v65, v46, v47);
          }
          while ( v60 != v43 );
        }
        ++v61;
        v59 += a4;
      }
      while ( a3 != v61 );
    }
    if ( v76 != (char *)&v77 + 8 )
      _libc_free((unsigned __int64)v76);
    result = *((_QWORD *)&v73 + 1);
    if ( *((_QWORD *)&v73 + 1) != -8 && *((_QWORD *)&v73 + 1) != 0 && *((_QWORD *)&v73 + 1) != -16 )
      return sub_1649B30(&v72);
  }
  else
  {
    if ( v76 != (char *)&v77 + 8 )
      _libc_free((unsigned __int64)v76);
    return sub_1455FA0((__int64)&v72);
  }
  return result;
}

// Function: sub_AD4210
// Address: 0xad4210
//
__int64 __fastcall sub_AD4210(__int64 a1, __int64 a2, __int16 *a3)
{
  __int16 v5; // ax
  bool v6; // zf
  __int64 v7; // rax
  int v8; // eax
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  int v11; // r13d
  __int64 v12; // r9
  int v13; // r13d
  int v14; // r11d
  unsigned int i; // ecx
  __int64 *v16; // r10
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned int v19; // ecx
  __int16 v20; // ax
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v26; // rdi
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  unsigned int v29; // esi
  int v30; // eax
  __int64 *v31; // rdx
  int v32; // eax
  const void *v33; // rax
  void *v34; // rdx
  int v35; // eax
  __int64 v36; // rax
  char v37; // al
  char v38; // al
  __int64 *v39; // [rsp+0h] [rbp-1B0h]
  __int64 *v40; // [rsp+8h] [rbp-1A8h]
  int v41; // [rsp+8h] [rbp-1A8h]
  int v42; // [rsp+10h] [rbp-1A0h]
  unsigned int v43; // [rsp+10h] [rbp-1A0h]
  __int64 *v44; // [rsp+10h] [rbp-1A0h]
  unsigned int v45; // [rsp+18h] [rbp-198h]
  __int64 v46; // [rsp+18h] [rbp-198h]
  int v47; // [rsp+18h] [rbp-198h]
  __int64 v48; // [rsp+20h] [rbp-190h]
  __int64 v49; // [rsp+20h] [rbp-190h]
  unsigned int v50; // [rsp+20h] [rbp-190h]
  __int64 v51; // [rsp+28h] [rbp-188h]
  __int64 v52; // [rsp+28h] [rbp-188h]
  __int64 v53; // [rsp+28h] [rbp-188h]
  __int64 *v54; // [rsp+40h] [rbp-170h] BYREF
  __int64 *v55; // [rsp+48h] [rbp-168h] BYREF
  unsigned __int64 v56; // [rsp+50h] [rbp-160h] BYREF
  __m128i v57; // [rsp+58h] [rbp-158h] BYREF
  __m128i v58; // [rsp+68h] [rbp-148h]
  __int64 v59; // [rsp+78h] [rbp-138h]
  __int64 v60; // [rsp+80h] [rbp-130h] BYREF
  unsigned int v61; // [rsp+88h] [rbp-128h]
  __int64 v62; // [rsp+90h] [rbp-120h] BYREF
  unsigned int v63; // [rsp+98h] [rbp-118h]
  char v64; // [rsp+A0h] [rbp-110h]
  __int64 v65; // [rsp+B0h] [rbp-100h] BYREF
  char v66[8]; // [rsp+B8h] [rbp-F8h] BYREF
  __m128i v67; // [rsp+C0h] [rbp-F0h] BYREF
  __m128i v68; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v69; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v70; // [rsp+E8h] [rbp-C8h] BYREF
  unsigned int v71; // [rsp+F0h] [rbp-C0h]
  __int64 v72; // [rsp+F8h] [rbp-B8h] BYREF
  unsigned int v73; // [rsp+100h] [rbp-B0h]
  char v74; // [rsp+108h] [rbp-A8h]
  unsigned __int64 v75; // [rsp+110h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+118h] [rbp-98h]
  __int16 v77; // [rsp+120h] [rbp-90h]
  __m128i v78; // [rsp+128h] [rbp-88h]
  void *s1[2]; // [rsp+138h] [rbp-78h]
  __int64 v80; // [rsp+148h] [rbp-68h]
  __int64 v81; // [rsp+150h] [rbp-60h] BYREF
  unsigned int v82; // [rsp+158h] [rbp-58h]
  __int64 v83; // [rsp+160h] [rbp-50h] BYREF
  unsigned int v84; // [rsp+168h] [rbp-48h]
  char v85; // [rsp+170h] [rbp-40h]

  v5 = *a3;
  v6 = *((_BYTE *)a3 + 80) == 0;
  v65 = a2;
  *(_WORD *)v66 = v5;
  v7 = *((_QWORD *)a3 + 5);
  v74 = 0;
  v69 = v7;
  v67 = _mm_loadu_si128((const __m128i *)(a3 + 4));
  v68 = _mm_loadu_si128((const __m128i *)(a3 + 12));
  if ( !v6 )
  {
    v71 = *((_DWORD *)a3 + 14);
    if ( v71 > 0x40 )
      sub_C43780(&v70, a3 + 24);
    else
      v70 = *((_QWORD *)a3 + 6);
    v73 = *((_DWORD *)a3 + 18);
    if ( v73 > 0x40 )
      sub_C43780(&v72, a3 + 32);
    else
      v72 = *((_QWORD *)a3 + 8);
    v74 = 1;
  }
  v75 = sub_AC61D0((__int64 *)v68.m128i_i64[0], v68.m128i_i64[0] + 4 * v68.m128i_i64[1]);
  v56 = sub_AC5F60((__int64 *)v67.m128i_i64[0], v67.m128i_i64[0] + 8 * v67.m128i_i64[1]);
  LODWORD(v75) = sub_AC5EC0(v66, &v66[1], (__int64 *)&v56, (__int64 *)&v75, &v69);
  v8 = sub_AC7AE0(&v65, &v75);
  v85 = 0;
  LODWORD(v75) = v8;
  v9 = _mm_loadu_si128(&v67);
  v10 = _mm_loadu_si128(&v68);
  v76 = v65;
  v78 = v9;
  v77 = *(_WORD *)v66;
  *(__m128i *)s1 = v10;
  v80 = v69;
  if ( v74 )
  {
    v82 = v71;
    if ( v71 > 0x40 )
      sub_C43780(&v81, &v70);
    else
      v81 = v70;
    v84 = v73;
    if ( v73 > 0x40 )
      sub_C43780(&v83, &v72);
    else
      v83 = v72;
    v11 = *(_DWORD *)(a1 + 24);
    v12 = *(_QWORD *)(a1 + 8);
    v85 = 1;
    if ( v11 )
      goto LABEL_4;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v12 = *(_QWORD *)(a1 + 8);
    if ( v11 )
    {
LABEL_4:
      v13 = v11 - 1;
      v14 = 1;
      for ( i = v13 & v75; ; i = v13 & v19 )
      {
        v16 = (__int64 *)(v12 + 8LL * i);
        v17 = *v16;
        v18 = *v16;
        if ( *v16 == -4096 )
          break;
        if ( v17 == -8192 )
          goto LABEL_72;
        if ( v76 != *(_QWORD *)(v17 + 8) )
          goto LABEL_8;
        if ( (unsigned __int8)v77 != *(unsigned __int16 *)(v17 + 2) )
          goto LABEL_8;
        if ( HIBYTE(v77) != *(_BYTE *)(v17 + 1) >> 1 )
          goto LABEL_8;
        v26 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
        if ( v78.m128i_i64[1] != v26 )
          goto LABEL_8;
        if ( (*(_DWORD *)(v17 + 4) & 0x7FFFFFF) != 0 )
        {
          v27 = (_QWORD *)v78.m128i_i64[0];
          v28 = (_QWORD *)(v17 - 32 * v26);
          while ( *v27 == *v28 )
          {
            ++v27;
            v28 += 4;
            if ( v27 == (_QWORD *)(v78.m128i_i64[0] + 8LL * ((*(_DWORD *)(v17 + 4) & 0x7FFFFFFu) - 1) + 8) )
              goto LABEL_69;
          }
          goto LABEL_8;
        }
LABEL_69:
        if ( *(_WORD *)(v17 + 2) == 63 )
        {
          v40 = (__int64 *)(v12 + 8LL * i);
          v42 = v14;
          v45 = i;
          v48 = v12;
          v51 = *v16;
          v33 = (const void *)sub_AC35F0(v17);
          v17 = v51;
          v12 = v48;
          i = v45;
          v14 = v42;
          v16 = v40;
          if ( s1[1] != v34 )
          {
            v17 = *v40;
LABEL_71:
            v18 = v17;
            goto LABEL_72;
          }
          if ( 4 * (__int64)s1[1] )
          {
            v35 = memcmp(s1[0], v33, 4 * (__int64)s1[1]);
            v17 = v51;
            v12 = v48;
            i = v45;
            v14 = v42;
            v16 = v40;
            if ( v35 )
              goto LABEL_79;
          }
        }
        else if ( s1[1] )
        {
          goto LABEL_71;
        }
        if ( *(_WORD *)(v17 + 2) == 34 )
        {
          v39 = v16;
          v41 = v14;
          v43 = i;
          v46 = v12;
          v49 = v80;
          v52 = v17;
          v36 = sub_AC5180(v17);
          v12 = v46;
          i = v43;
          v14 = v41;
          v16 = v39;
          if ( v49 == v36 )
          {
            if ( *(_WORD *)(v52 + 2) == 34 )
            {
              sub_AC51A0((__int64)&v56, v52);
              v12 = v46;
              i = v43;
              v14 = v41;
              v16 = v39;
              if ( v85 )
              {
                if ( v58.m128i_i8[8] )
                {
                  if ( v82 == v57.m128i_i32[0] )
                  {
                    v37 = sub_AAD8B0((__int64)&v81, &v56);
                    v12 = v46;
                    i = v43;
                    v14 = v41;
                    v16 = v39;
                    if ( v37 )
                    {
                      v38 = sub_AAD8B0((__int64)&v83, &v57.m128i_i64[1]);
                      v12 = v46;
                      i = v43;
                      v14 = v41;
                      v16 = v39;
                      if ( v38 )
                      {
                        sub_9963D0((__int64)&v56);
                        v16 = v39;
LABEL_82:
                        if ( v16 == (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 24)) )
                          break;
                        v24 = *v16;
                        goto LABEL_17;
                      }
                    }
                  }
                  v44 = v16;
                  v47 = v14;
                  v50 = i;
                  v53 = v12;
                  sub_9963D0((__int64)&v56);
                  v12 = v53;
                  i = v50;
                  v14 = v47;
                  v18 = *v44;
                  goto LABEL_72;
                }
              }
              else
              {
                if ( !v58.m128i_i8[8] )
                  goto LABEL_82;
                sub_9963D0((__int64)&v56);
                v12 = v46;
                i = v43;
                v14 = v41;
                v16 = v39;
              }
              goto LABEL_79;
            }
LABEL_81:
            v58.m128i_i8[8] = 0;
            if ( !v85 )
              goto LABEL_82;
          }
        }
        else if ( !v80 )
        {
          goto LABEL_81;
        }
LABEL_79:
        v18 = *v16;
LABEL_72:
        if ( v18 == -4096 )
          break;
LABEL_8:
        v19 = v14 + i;
        ++v14;
      }
    }
  }
  v20 = *a3;
  v21 = _mm_loadu_si128((const __m128i *)(a3 + 4));
  v64 = 0;
  v22 = _mm_loadu_si128((const __m128i *)(a3 + 12));
  v6 = *((_BYTE *)a3 + 80) == 0;
  LOWORD(v56) = v20;
  v23 = *((_QWORD *)a3 + 5);
  v57 = v21;
  v59 = v23;
  v58 = v22;
  if ( !v6 )
  {
    v61 = *((_DWORD *)a3 + 14);
    if ( v61 > 0x40 )
      sub_C43780(&v60, a3 + 24);
    else
      v60 = *((_QWORD *)a3 + 6);
    v63 = *((_DWORD *)a3 + 18);
    if ( v63 > 0x40 )
      sub_C43780(&v62, a3 + 32);
    else
      v62 = *((_QWORD *)a3 + 8);
    v64 = 1;
  }
  v24 = sub_AC4A80((unsigned __int8 *)&v56, a2);
  if ( !(unsigned __int8)sub_AC8350(a1, (__int64)&v75, &v54) )
  {
    v29 = *(_DWORD *)(a1 + 24);
    v30 = *(_DWORD *)(a1 + 16);
    v31 = v54;
    ++*(_QWORD *)a1;
    v32 = v30 + 1;
    v55 = v31;
    if ( 4 * v32 >= 3 * v29 )
    {
      v29 *= 2;
    }
    else if ( v29 - *(_DWORD *)(a1 + 20) - v32 > v29 >> 3 )
    {
LABEL_57:
      *(_DWORD *)(a1 + 16) = v32;
      if ( *v31 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v31 = v24;
      goto LABEL_16;
    }
    sub_AD4030(a1, v29);
    sub_AC8350(a1, (__int64)&v75, &v55);
    v31 = v55;
    v32 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_57;
  }
LABEL_16:
  if ( v64 )
  {
    v64 = 0;
    if ( v63 > 0x40 && v62 )
      j_j___libc_free_0_0(v62);
    if ( v61 > 0x40 && v60 )
      j_j___libc_free_0_0(v60);
  }
LABEL_17:
  if ( v85 )
  {
    v85 = 0;
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    if ( v82 > 0x40 && v81 )
      j_j___libc_free_0_0(v81);
  }
  if ( v74 )
  {
    v74 = 0;
    if ( v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
    if ( v71 > 0x40 && v70 )
      j_j___libc_free_0_0(v70);
  }
  return v24;
}

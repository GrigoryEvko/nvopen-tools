// Function: sub_BEEBD0
// Address: 0xbeebd0
//
void __fastcall sub_BEEBD0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // r11
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 *v12; // rdi
  __int64 *v13; // r10
  __int64 v14; // rdx
  __int64 v15; // rax
  __int16 v16; // ax
  const char *v17; // rax
  __int64 v18; // r13
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // r11
  unsigned __int8 *v24; // r11
  unsigned __int8 *v25; // rdx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  const char *v30; // rax
  const char *v31; // rax
  __int64 v32; // r15
  char *v33; // r13
  int v34; // r12d
  __int64 i; // rax
  unsigned int v36; // eax
  __int64 v37; // rdx
  int v38; // r14d
  __int64 j; // rax
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned int v45; // r13d
  __int64 v46; // rax
  __int64 v47; // rax
  __m128i *v48; // rsi
  __m128i *v49; // r13
  __int64 v50; // [rsp+20h] [rbp-1C0h]
  __int64 v51; // [rsp+38h] [rbp-1A8h]
  __int64 v52; // [rsp+40h] [rbp-1A0h]
  int v53; // [rsp+40h] [rbp-1A0h]
  int v54; // [rsp+50h] [rbp-190h]
  __int64 v55; // [rsp+58h] [rbp-188h]
  int v56; // [rsp+58h] [rbp-188h]
  __m128i v57[2]; // [rsp+60h] [rbp-180h] BYREF
  char v58; // [rsp+80h] [rbp-160h]
  char v59; // [rsp+81h] [rbp-15Fh]
  __m128i v60; // [rsp+90h] [rbp-150h] BYREF
  __int16 v61; // [rsp+B0h] [rbp-130h]
  __m128i v62[2]; // [rsp+C0h] [rbp-120h] BYREF
  char v63; // [rsp+E0h] [rbp-100h]
  char v64; // [rsp+E1h] [rbp-FFh]
  __m128i v65; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v66; // [rsp+100h] [rbp-E0h]
  _DWORD v67[2]; // [rsp+108h] [rbp-D8h] BYREF
  char v68; // [rsp+110h] [rbp-D0h]
  char v69; // [rsp+111h] [rbp-CFh]
  __m128i v70; // [rsp+150h] [rbp-90h] BYREF
  char v71; // [rsp+168h] [rbp-78h] BYREF
  char v72; // [rsp+170h] [rbp-70h]
  char v73; // [rsp+171h] [rbp-6Fh]

  v7 = (__int64)a1;
  v8 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v8 == 25 )
  {
    v73 = 1;
    v17 = "cannot use musttail call with inline asm";
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_QWORD *)(a2 + 80);
    v11 = *(_QWORD *)(v9 + 72);
    v55 = *(_QWORD *)(v11 + 24);
    if ( (*(_DWORD *)(v55 + 8) >> 8 != 0) != (*(_DWORD *)(v10 + 8) >> 8 != 0) )
    {
      v73 = 1;
      v70.m128i_i64[0] = (__int64)"cannot guarantee tail call due to mismatched varargs";
      v72 = 3;
      sub_BDBF70(a1, (__int64)&v70);
      if ( !*a1 )
        return;
      goto LABEL_15;
    }
    v12 = *(__int64 **)(v10 + 16);
    v13 = *(__int64 **)(v55 + 16);
    v14 = *v12;
    v15 = *v13;
    if ( *v12 == *v13
      || *(_BYTE *)(v15 + 8) == 14
      && *(_BYTE *)(v14 + 8) == 14
      && *(_DWORD *)(v15 + 8) >> 8 == *(_DWORD *)(v14 + 8) >> 8 )
    {
      v16 = (*(_WORD *)(a2 + 2) >> 2) & 0x3FF;
      if ( ((*(_WORD *)(v11 + 2) >> 4) & 0x3FF) != v16 )
      {
        v73 = 1;
        v17 = "cannot guarantee tail call due to mismatched calling conv";
        goto LABEL_9;
      }
      v21 = *(_QWORD *)(a2 + 32);
      if ( !v21 || v21 == v9 + 48 )
        goto LABEL_44;
      v22 = v21 - 24;
      if ( *(_BYTE *)(v21 - 24) == 78 )
      {
        v23 = *(_QWORD *)(v21 - 56);
        if ( !v23 || a2 != v23 )
        {
          v73 = 1;
          v31 = "bitcast following musttail call must use the call";
          goto LABEL_50;
        }
        a5 = *(_QWORD *)(v21 + 8);
        if ( a5 == *(_QWORD *)(v21 + 16) + 48LL || !a5 )
        {
LABEL_44:
          v65.m128i_i64[0] = a2;
          v30 = "musttail call must precede a ret with an optional bitcast";
          v73 = 1;
          goto LABEL_45;
        }
        v24 = (unsigned __int8 *)(v21 - 24);
        v22 = a5 - 24;
      }
      else
      {
        v24 = (unsigned __int8 *)a2;
      }
      if ( *(_BYTE *)v22 == 30 )
      {
        if ( (*(_DWORD *)(v22 + 4) & 0x7FFFFFF) == 0
          || (v25 = *(unsigned __int8 **)(v22 - 32LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))) == 0
          || v25 == v24
          || (unsigned int)*v25 - 12 <= 1 )
        {
          v52 = *(_QWORD *)(v11 + 120);
          v51 = *(_QWORD *)(a2 + 72);
          if ( v16 == 20 )
          {
            v32 = 11;
            v33 = "swifttailcc";
          }
          else
          {
            if ( v16 != 18 )
            {
              v26 = *(_DWORD *)(v55 + 12);
              v56 = v26 - 1;
              if ( !*(_BYTE *)v8 && v10 == *(_QWORD *)(v8 + 24) && (*(_BYTE *)(v8 + 33) & 0x20) != 0 )
              {
                if ( v26 != 1 )
                {
LABEL_89:
                  v45 = 0;
                  while ( 1 )
                  {
                    v46 = sub_B2BE50(v11);
                    sub_BD8DD0((__int64)&v65, v46, v45, v52);
                    v47 = sub_B2BE50(v11);
                    sub_BD8DD0((__int64)&v70, v47, v45, v51);
                    if ( !(unsigned __int8)sub_A75080((__int64)&v65, (__int64)&v70) )
                      break;
                    if ( (char *)v70.m128i_i64[1] != &v71 )
                      _libc_free(v70.m128i_i64[1], &v70);
                    if ( (_DWORD *)v65.m128i_i64[1] != v67 )
                      _libc_free(v65.m128i_i64[1], &v70);
                    if ( ++v45 == v56 )
                      return;
                  }
                  v64 = 1;
                  v63 = 3;
                  v48 = v62;
                  v49 = *(__m128i **)(a2 + 32 * (v45 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
                  v62[0].m128i_i64[0] = (__int64)"cannot guarantee tail call due to mismatched ABI impacting function attributes";
                  if ( v49 )
                  {
                    sub_BDBF70((__int64 *)v7, (__int64)v62);
                    if ( *(_QWORD *)v7 )
                    {
                      sub_BDBD80(v7, (_BYTE *)a2);
                      v48 = v49;
                      sub_BDBD80(v7, v49);
                    }
                  }
                  else
                  {
                    sub_BDBF70((__int64 *)v7, (__int64)v62);
                    if ( *(_QWORD *)v7 )
                    {
                      v48 = (__m128i *)a2;
                      sub_BDBD80(v7, (_BYTE *)a2);
                    }
                  }
                  if ( (char *)v70.m128i_i64[1] != &v71 )
                    _libc_free(v70.m128i_i64[1], v48);
                  if ( (_DWORD *)v65.m128i_i64[1] != v67 )
                    _libc_free(v65.m128i_i64[1], v48);
                }
                return;
              }
              if ( *(_DWORD *)(v10 + 12) == v26 )
              {
                if ( v26 == 1 )
                  return;
                v27 = 0;
                while ( 1 )
                {
                  v28 = v12[v27 + 1];
                  v29 = v13[v27 + 1];
                  if ( v29 != v28
                    && (*(_BYTE *)(v29 + 8) != 14
                     || *(_BYTE *)(v28 + 8) != 14
                     || *(_DWORD *)(v29 + 8) >> 8 != *(_DWORD *)(v28 + 8) >> 8) )
                  {
                    break;
                  }
                  if ( v56 == ++v27 )
                    goto LABEL_89;
                }
                v65.m128i_i64[0] = a2;
                v30 = "cannot guarantee tail call due to mismatched parameter types";
                v73 = 1;
              }
              else
              {
                v65.m128i_i64[0] = a2;
                v30 = "cannot guarantee tail call due to mismatched parameter counts";
                v73 = 1;
              }
LABEL_45:
              v72 = 3;
              v70.m128i_i64[0] = (__int64)v30;
              sub_BEEAD0((_BYTE *)v7, (__int64)&v70, &v65);
              return;
            }
            v32 = 6;
            v33 = "tailcc";
          }
          v54 = *(_DWORD *)(v55 + 12);
          if ( v54 != 1 )
          {
            v50 = v7;
            v34 = 0;
            for ( i = sub_B2BE50(v11); ; i = sub_B2BE50(v11) )
            {
              sub_BD8DD0((__int64)&v70, i, v34, v52);
              v65.m128i_i64[0] = (__int64)v67;
              v66 = 32;
              v65.m128i_i64[1] = v32 + 16;
              if ( (unsigned int)v32 < 8 )
              {
                v67[0] = *(_DWORD *)v33;
                *(_DWORD *)((char *)&v67[-1] + (unsigned int)v32) = *(_DWORD *)&v33[(unsigned int)v32 - 4];
              }
              else
              {
                *(_QWORD *)((char *)&v67[-2] + v32) = *(_QWORD *)&v33[v32 - 8];
                if ( (unsigned int)(v32 - 1) >= 8 )
                {
                  v36 = 0;
                  do
                  {
                    v37 = v36;
                    v36 += 8;
                    *(_QWORD *)((char *)v67 + v37) = *(_QWORD *)&v33[v37];
                  }
                  while ( v36 < (((_DWORD)v32 - 1) & 0xFFFFFFF8) );
                }
              }
              *(__m128i *)((char *)v67 + v32) = _mm_load_si128((const __m128i *)&xmmword_3F64260);
              sub_BDBFE0(v50, (__int64)&v70, v65.m128i_i64[0], v65.m128i_i64[1]);
              if ( (_DWORD *)v65.m128i_i64[0] != v67 )
                _libc_free(v65.m128i_i64[0], &v70);
              if ( (char *)v70.m128i_i64[1] != &v71 )
                _libc_free(v70.m128i_i64[1], &v70);
              if ( ++v34 == v54 - 1 )
                break;
            }
            v7 = v50;
          }
          v53 = *(_DWORD *)(v10 + 12) - 1;
          if ( *(_DWORD *)(v10 + 12) != 1 )
          {
            v38 = 0;
            for ( j = sub_B2BE50(v11); ; j = sub_B2BE50(v11) )
            {
              sub_BD8DD0((__int64)&v70, j, v38, v51);
              v65.m128i_i64[0] = (__int64)v67;
              v66 = 32;
              v65.m128i_i64[1] = v32 + 16;
              if ( (unsigned int)v32 < 8 )
              {
                v67[0] = *(_DWORD *)v33;
                *(_DWORD *)((char *)&v67[-1] + (unsigned int)v32) = *(_DWORD *)&v33[(unsigned int)v32 - 4];
              }
              else
              {
                *(_QWORD *)((char *)&v67[-2] + v32) = *(_QWORD *)&v33[v32 - 8];
                if ( (unsigned int)(v32 - 1) >= 8 )
                {
                  v40 = 0;
                  do
                  {
                    v41 = v40;
                    v40 += 8;
                    *(_QWORD *)((char *)v67 + v41) = *(_QWORD *)&v33[v41];
                  }
                  while ( v40 < (((_DWORD)v32 - 1) & 0xFFFFFFF8) );
                }
              }
              *(__m128i *)((char *)v67 + v32) = _mm_load_si128((const __m128i *)&xmmword_3F64270);
              sub_BDBFE0(v7, (__int64)&v70, v65.m128i_i64[0], v65.m128i_i64[1]);
              if ( (_DWORD *)v65.m128i_i64[0] != v67 )
                _libc_free(v65.m128i_i64[0], &v70);
              if ( (char *)v70.m128i_i64[1] != &v71 )
                _libc_free(v70.m128i_i64[1], &v70);
              if ( ++v38 == v53 )
                break;
            }
          }
          if ( *(_DWORD *)(v55 + 8) >> 8 )
          {
            v60.m128i_i64[0] = (__int64)v33;
            v65.m128i_i64[0] = (__int64)" tail call for varargs function";
            v61 = 261;
            v57[0].m128i_i64[0] = (__int64)"cannot guarantee ";
            v69 = 1;
            v68 = 3;
            v60.m128i_i64[1] = v32;
            v59 = 1;
            v58 = 3;
            sub_9C6370(v62, v57, &v60, 261, a5, a6);
            sub_9C6370(&v70, v62, &v65, v42, v43, v44);
            sub_BDBF70((__int64 *)v7, (__int64)&v70);
          }
          return;
        }
        v73 = 1;
        v31 = "musttail call result must be returned";
LABEL_50:
        v70.m128i_i64[0] = (__int64)v31;
        v72 = 3;
        sub_BDBF70((__int64 *)v7, (__int64)&v70);
        if ( *(_QWORD *)v7 )
          sub_BDBD80(v7, (_BYTE *)v22);
        return;
      }
      goto LABEL_44;
    }
    v73 = 1;
    v17 = "cannot guarantee tail call due to mismatched return types";
  }
LABEL_9:
  v18 = *(_QWORD *)v7;
  v70.m128i_i64[0] = (__int64)v17;
  v72 = 3;
  if ( !v18 )
  {
    *(_BYTE *)(v7 + 152) = 1;
    return;
  }
  sub_CA0E80(&v70, v18);
  v19 = *(_BYTE **)(v18 + 32);
  if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
  {
    sub_CB5D20(v18, 10);
  }
  else
  {
    *(_QWORD *)(v18 + 32) = v19 + 1;
    *v19 = 10;
  }
  v20 = *(_QWORD *)v7;
  *(_BYTE *)(v7 + 152) = 1;
  if ( v20 )
LABEL_15:
    sub_BDBD80(v7, (_BYTE *)a2);
}

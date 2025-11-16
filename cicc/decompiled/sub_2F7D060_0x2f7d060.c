// Function: sub_2F7D060
// Address: 0x2f7d060
//
__int64 __fastcall sub_2F7D060(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __m128i *v8; // rdx
  __int128 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned int v12; // ebx
  int v13; // esi
  unsigned int i; // edx
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 *v23; // rbx
  __int64 *v24; // r12
  int v25; // eax
  __int64 v26; // rdi
  __int16 v28; // cx
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rbx
  bool v32; // zf
  __m128i *v33; // rax
  int v34; // esi
  __int64 v35; // rdx
  __int64 *v36; // rax
  __int64 v37; // r14
  __int64 v38; // rbx
  int v39; // eax
  int v40; // r9d
  unsigned int v41; // edx
  __int64 v42; // rax
  unsigned int v43; // edx
  char v44; // r10
  __int64 v45; // rax
  unsigned int v46; // [rsp+1Ch] [rbp-104h]
  __int64 v47; // [rsp+20h] [rbp-100h]
  __m128i *v48; // [rsp+28h] [rbp-F8h]
  __int64 v49; // [rsp+28h] [rbp-F8h]
  int v50; // [rsp+3Ch] [rbp-E4h] BYREF
  __m128i *v51; // [rsp+40h] [rbp-E0h] BYREF
  __m128i *v52; // [rsp+48h] [rbp-D8h] BYREF
  __int128 v53; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+60h] [rbp-C0h]
  unsigned int v55; // [rsp+68h] [rbp-B8h]
  __int128 *v56; // [rsp+70h] [rbp-B0h] BYREF
  __int128 v57; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v58; // [rsp+88h] [rbp-98h]
  __int64 *v59; // [rsp+90h] [rbp-90h]
  __int64 *v60; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-78h]
  _BYTE v62[112]; // [rsp+B0h] [rbp-70h] BYREF

  v60 = (__int64 *)v62;
  v61 = 0x800000000LL;
  v2 = *(_QWORD *)(a1 + 32);
  v54 = 0;
  v55 = 0;
  v3 = *(_QWORD *)(v2 + 16);
  v53 = 0u;
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v47 = a1 + 48;
  if ( *(_QWORD *)(a1 + 56) == a1 + 48 )
    goto LABEL_21;
  v5 = *(_QWORD *)(a1 + 56);
  do
  {
    if ( (unsigned __int16)(*(_WORD *)(v5 + 68) - 14) > 1u )
      goto LABEL_18;
    v6 = sub_B10CD0(v5 + 56);
    v7 = *(_BYTE *)(v6 - 16);
    if ( (v7 & 2) != 0 )
    {
      if ( *(_DWORD *)(v6 - 24) != 2 )
        goto LABEL_6;
      v17 = *(_QWORD *)(v6 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 2 )
      {
LABEL_6:
        v8 = 0;
        goto LABEL_7;
      }
      v17 = v6 - 16 - 8LL * ((v7 >> 2) & 0xF);
    }
    v8 = *(__m128i **)(v17 + 8);
LABEL_7:
    v48 = v8;
    v9 = (__int128 *)sub_2E89170(v5);
    v10 = v55;
    LOBYTE(v58) = 0;
    v56 = v9;
    v11 = *((_QWORD *)&v53 + 1);
    v59 = (__int64 *)v48;
    if ( !v55 )
    {
LABEL_35:
      v28 = *(_WORD *)(v5 + 68);
      v15 = v11 + 56 * v10;
      v29 = v15;
      if ( v28 == 15 )
        goto LABEL_37;
LABEL_36:
      if ( v28 != 14 )
        goto LABEL_37;
      v30 = *(_QWORD *)(v5 + 32);
      if ( *(_BYTE *)v30 )
        goto LABEL_48;
LABEL_38:
      if ( v29 != v15 && *(_DWORD *)(*(_QWORD *)(v15 + 40) + 8LL) == *(_DWORD *)(v30 + 8) )
      {
        v18 = *(_QWORD *)(v15 + 48);
        if ( v18 == sub_2E891C0(v5) )
        {
          v21 = (unsigned int)v61;
          v22 = (unsigned int)v61 + 1LL;
          if ( v22 > HIDWORD(v61) )
          {
            sub_C8D5F0((__int64)&v60, v62, v22, 8u, v19, v20);
            v21 = (unsigned int)v61;
          }
          v60[v21] = v5;
          LODWORD(v61) = v61 + 1;
LABEL_18:
          if ( (*(_BYTE *)(*(_QWORD *)(v5 + 16) + 24LL) & 0x10) == 0 && (_DWORD)v54 )
          {
            v58 = *((_QWORD *)&v53 + 1) + 56LL * v55;
            v57 = v53;
            v56 = &v53;
            sub_2F7CE20((__int64)&v56);
            v37 = *((_QWORD *)&v53 + 1) + 56LL * v55;
            while ( 1 )
            {
              v38 = *((_QWORD *)&v57 + 1);
              do
              {
LABEL_53:
                if ( v37 == v38 )
                  goto LABEL_19;
                if ( (unsigned int)sub_2E8E710(v5, *(_DWORD *)(*(_QWORD *)(v38 + 40) + 8LL), v4, 0, 1) != -1 && v55 )
                {
                  v50 = 0;
                  if ( *(_BYTE *)(v38 + 24) )
                    v50 = *(unsigned __int16 *)(v38 + 16) | (*(_DWORD *)(v38 + 8) << 16);
                  v46 = v55;
                  v52 = *(__m128i **)(v38 + 32);
                  v49 = *((_QWORD *)&v53 + 1);
                  v51 = *(__m128i **)v38;
                  v39 = sub_F11290((__int64 *)&v51, &v50, (__int64 *)&v52);
                  v40 = 1;
                  v41 = (v46 - 1) & v39;
                  while ( 2 )
                  {
                    v42 = v49 + 56LL * v41;
                    if ( *(_QWORD *)v42 == *(_QWORD *)v38 && (v44 = *(_BYTE *)(v38 + 24), v44 == *(_BYTE *)(v42 + 24)) )
                    {
                      if ( !v44
                        || *(_QWORD *)(v38 + 8) == *(_QWORD *)(v42 + 8)
                        && *(_QWORD *)(v38 + 16) == *(_QWORD *)(v42 + 16) )
                      {
                        if ( *(_QWORD *)(v38 + 32) == *(_QWORD *)(v42 + 32) )
                        {
                          *(_QWORD *)v42 = 0;
                          *(_QWORD *)(v42 + 8) = 0;
                          *(_QWORD *)(v42 + 16) = 0;
                          LODWORD(v54) = v54 - 1;
                          ++HIDWORD(v54);
                          *(_BYTE *)(v42 + 24) = 1;
                          *(_QWORD *)(v42 + 32) = 0;
                          break;
                        }
                        goto LABEL_60;
                      }
                    }
                    else
                    {
LABEL_60:
                      if ( !*(_QWORD *)v42 && !*(_BYTE *)(v42 + 24) && !*(_QWORD *)(v42 + 32) )
                        break;
                    }
                    v43 = v40 + v41;
                    ++v40;
                    v41 = (v46 - 1) & v43;
                    continue;
                  }
                }
                v38 = v58;
                v45 = *((_QWORD *)&v57 + 1) + 56LL;
                *((_QWORD *)&v57 + 1) = v45;
              }
              while ( v45 == v58 );
              while ( !*(_QWORD *)v45
                   && (!*(_BYTE *)(v45 + 24) || !*(_QWORD *)(v45 + 8) && !*(_QWORD *)(v45 + 16))
                   && !*(_QWORD *)(v45 + 32) )
              {
                v45 += 56;
                *((_QWORD *)&v57 + 1) = v45;
                if ( v45 == v58 )
                  goto LABEL_53;
              }
            }
          }
          goto LABEL_19;
        }
      }
      v31 = sub_2E891C0(v5);
      v32 = (unsigned __int8)sub_2F7CCC0((__int64)&v53, (__int64)&v56, &v51) == 0;
      v33 = v51;
      if ( !v32 )
      {
LABEL_46:
        v36 = &v33[2].m128i_i64[1];
        *v36 = v30;
        v36[1] = v31;
        goto LABEL_19;
      }
      v34 = v55;
      v52 = v51;
      *(_QWORD *)&v53 = v53 + 1;
      if ( 4 * ((int)v54 + 1) >= 3 * v55 )
      {
        v34 = 2 * v55;
      }
      else if ( v55 - HIDWORD(v54) - ((_DWORD)v54 + 1) > v55 >> 3 )
      {
        LODWORD(v54) = v54 + 1;
        if ( v51->m128i_i64[0] )
        {
LABEL_44:
          --HIDWORD(v54);
LABEL_45:
          *v33 = _mm_loadu_si128((const __m128i *)&v56);
          v33[1] = _mm_loadu_si128((const __m128i *)((char *)&v57 + 8));
          v35 = (__int64)v59;
          v33[2].m128i_i64[1] = 0;
          v33[2].m128i_i64[0] = v35;
          v33[3].m128i_i64[0] = 0;
          goto LABEL_46;
        }
LABEL_82:
        if ( !v33[1].m128i_i8[8] && !v33[2].m128i_i64[0] )
          goto LABEL_45;
        goto LABEL_44;
      }
      sub_2F7CE70((__int64)&v53, v34);
      sub_2F7CCC0((__int64)&v53, (__int64)&v56, &v52);
      v33 = v52;
      LODWORD(v54) = v54 + 1;
      if ( v52->m128i_i64[0] )
        goto LABEL_44;
      goto LABEL_82;
    }
    v52 = v48;
    v12 = v55 - 1;
    v51 = (__m128i *)v9;
    v50 = 0;
    v13 = 1;
    for ( i = v12 & sub_F11290((__int64 *)&v51, &v50, (__int64 *)&v52); ; i = v12 & v16 )
    {
      v15 = v11 + 56LL * i;
      if ( *(__int128 **)v15 == v56 && (_BYTE)v58 == *(_BYTE *)(v15 + 24) )
      {
        if ( (_BYTE)v58 && v57 != *(_OWORD *)(v15 + 8) )
          goto LABEL_11;
        if ( v59 == *(__int64 **)(v15 + 32) )
          break;
      }
      if ( !*(_QWORD *)v15 && !*(_BYTE *)(v15 + 24) && !*(_QWORD *)(v15 + 32) )
      {
        v11 = *((_QWORD *)&v53 + 1);
        v10 = v55;
        goto LABEL_35;
      }
LABEL_11:
      v16 = v13 + i;
      ++v13;
    }
    v28 = *(_WORD *)(v5 + 68);
    v29 = *((_QWORD *)&v53 + 1) + 56LL * v55;
    if ( v28 != 15 )
      goto LABEL_36;
    if ( v15 != v29 )
      goto LABEL_49;
LABEL_37:
    v30 = *(_QWORD *)(v5 + 32) + 80LL;
    if ( !*(_BYTE *)v30 )
      goto LABEL_38;
LABEL_48:
    if ( v29 != v15 )
    {
LABEL_49:
      *(_QWORD *)v15 = 0;
      *(_QWORD *)(v15 + 8) = 0;
      *(_QWORD *)(v15 + 16) = 0;
      *(_BYTE *)(v15 + 24) = 1;
      *(_QWORD *)(v15 + 32) = 0;
      LODWORD(v54) = v54 - 1;
      ++HIDWORD(v54);
    }
LABEL_19:
    if ( (*(_BYTE *)v5 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
        v5 = *(_QWORD *)(v5 + 8);
    }
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v47 != v5 );
LABEL_21:
  v23 = v60;
  v24 = &v60[(unsigned int)v61];
  v25 = v61;
  if ( v24 != v60 )
  {
    do
    {
      v26 = *v23++;
      sub_2E88E20(v26);
    }
    while ( v24 != v23 );
    v25 = v61;
  }
  LOBYTE(v24) = v25 != 0;
  sub_C7D6A0(*((__int64 *)&v53 + 1), 56LL * v55, 8);
  if ( v60 != (__int64 *)v62 )
    _libc_free((unsigned __int64)v60);
  return (unsigned int)v24;
}

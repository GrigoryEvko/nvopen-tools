// Function: sub_206CB20
// Address: 0x206cb20
//
__int64 *__fastcall sub_206CB20(__int64 a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 **v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  const void **v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // eax
  char v21; // r8
  unsigned int v22; // r15d
  unsigned int v23; // ecx
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 *v27; // r11
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // r10
  unsigned __int64 v31; // rcx
  bool v32; // r9
  bool v33; // dl
  bool v34; // al
  __int64 *v35; // r14
  unsigned __int64 v36; // r9
  int v37; // edx
  const void **v38; // r8
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 *v42; // r14
  int v43; // edx
  int v44; // r15d
  __int64 *result; // rax
  unsigned int v46; // eax
  __int64 v47; // rax
  char v48; // dl
  unsigned __int64 v49; // rax
  void *v50; // rdx
  int v51; // ecx
  char v52; // dl
  __int64 v53; // r14
  char v54; // di
  __int64 v55; // rax
  int v56; // eax
  __int64 *v57; // r10
  __int64 *v58; // r11
  unsigned int v59; // eax
  __int64 v60; // rax
  unsigned int v61; // edx
  int v62; // edx
  __int128 v63; // [rsp-10h] [rbp-100h]
  __int64 v64; // [rsp+8h] [rbp-E8h]
  unsigned int v65; // [rsp+10h] [rbp-E0h]
  unsigned int v66; // [rsp+10h] [rbp-E0h]
  __int64 v67; // [rsp+10h] [rbp-E0h]
  __int64 v68; // [rsp+18h] [rbp-D8h]
  __int64 *v69; // [rsp+18h] [rbp-D8h]
  const void **v70; // [rsp+18h] [rbp-D8h]
  char v71; // [rsp+18h] [rbp-D8h]
  __int64 *v72; // [rsp+18h] [rbp-D8h]
  __int64 *v73; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v74; // [rsp+28h] [rbp-C8h]
  __int64 v76; // [rsp+38h] [rbp-B8h]
  __int64 *v77; // [rsp+38h] [rbp-B8h]
  __int128 v78; // [rsp+40h] [rbp-B0h]
  unsigned int v79; // [rsp+90h] [rbp-60h] BYREF
  const void **v80; // [rsp+98h] [rbp-58h]
  __int64 v81; // [rsp+A0h] [rbp-50h] BYREF
  int v82; // [rsp+A8h] [rbp-48h]
  __int64 v83; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v8 = *(__int64 ***)(a2 - 8);
  else
    v8 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v73 = sub_20685E0(a1, *v8, a4, a5, a6);
  v74 = v9;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v10 = *(_QWORD *)(a2 - 8);
  else
    v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  *(_QWORD *)&v78 = sub_20685E0(a1, *(__int64 **)(v10 + 24), a4, a5, a6);
  v12 = (unsigned int)v11;
  v76 = v78;
  v13 = *(_QWORD *)(a1 + 552);
  *((_QWORD *)&v78 + 1) = v11;
  v14 = 16LL * (unsigned int)v11;
  v15 = *(_QWORD *)(v13 + 16);
  v16 = sub_1E0A0C0(*(_QWORD *)(v13 + 32));
  v79 = sub_1F40B60(
          v15,
          *(unsigned __int8 *)(v14 + *(_QWORD *)(v76 + 40)),
          *(_QWORD *)(v14 + *(_QWORD *)(v76 + 40) + 8),
          v16,
          1);
  v17 = *(_QWORD *)a2;
  v80 = v18;
  if ( *(_BYTE *)(v17 + 8) != 16 )
  {
    v19 = v14 + *(_QWORD *)(v78 + 40);
    if ( (_BYTE)v79 != *(_BYTE *)v19 )
    {
      if ( (_BYTE)v79 )
      {
        v68 = *(_QWORD *)(v19 + 8);
        v20 = sub_2045180(v79);
        LOBYTE(v83) = v21;
        v22 = v20;
        v84 = v68;
        if ( v21 )
        {
LABEL_9:
          v23 = sub_2045180(v21);
LABEL_10:
          v24 = *(_DWORD *)(a1 + 536);
          v25 = *(_QWORD *)a1;
          v81 = 0;
          v82 = v24;
          if ( v25 )
          {
            if ( &v81 != (__int64 *)(v25 + 48) )
            {
              v26 = *(_QWORD *)(v25 + 48);
              v81 = v26;
              if ( v26 )
              {
                v65 = v23;
                sub_1623A60((__int64)&v81, v26, 2);
                v23 = v65;
              }
            }
          }
          v27 = *(__int64 **)(a1 + 552);
          if ( v22 <= v23 )
          {
            v53 = *(_QWORD *)(v78 + 40) + v14;
            v54 = *(_BYTE *)v53;
            v55 = *(_QWORD *)(v53 + 8);
            LOBYTE(v83) = v54;
            v84 = v55;
            if ( v54 )
            {
              v56 = sub_2045180(v54);
            }
            else
            {
              v77 = v27;
              v56 = sub_1F58D40((__int64)&v83);
              v57 = &v81;
              v58 = v77;
            }
            v59 = v56 - 1;
            if ( v59 )
            {
              _BitScanReverse(&v59, v59);
              if ( 32 - (v59 ^ 0x1F) > v22 )
              {
                v72 = v57;
                v60 = sub_1D323C0(
                        v58,
                        v78,
                        *((__int64 *)&v78 + 1),
                        (__int64)v57,
                        5,
                        0,
                        *(double *)a4.m128i_i64,
                        *(double *)a5.m128i_i64,
                        *(double *)a6.m128i_i64);
                v30 = (__int64)v72;
                LODWORD(v12) = v61;
                v76 = v60;
                *((_QWORD *)&v78 + 1) = v61 | *((_QWORD *)&v78 + 1) & 0xFFFFFFFF00000000LL;
LABEL_17:
                v12 = (unsigned int)v12;
                if ( v81 )
                  sub_161E7C0(v30, v81);
                goto LABEL_19;
              }
            }
            v69 = v57;
            v28 = sub_1D309E0(
                    v58,
                    145,
                    (__int64)v57,
                    v79,
                    v80,
                    0,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v78);
            LODWORD(v12) = v62;
          }
          else
          {
            v69 = &v81;
            v28 = sub_1D309E0(
                    v27,
                    143,
                    (__int64)&v81,
                    v79,
                    v80,
                    0,
                    *(double *)a4.m128i_i64,
                    *(double *)a5.m128i_i64,
                    *(double *)a6.m128i_i64,
                    v78);
            LODWORD(v12) = v29;
          }
          v30 = (__int64)v69;
          v76 = v28;
          *((_QWORD *)&v78 + 1) = (unsigned int)v12 | *((_QWORD *)&v78 + 1) & 0xFFFFFFFF00000000LL;
          goto LABEL_17;
        }
LABEL_37:
        v23 = sub_1F58D40((__int64)&v83);
        goto LABEL_10;
      }
LABEL_36:
      v67 = *(_QWORD *)(v19 + 8);
      v71 = *(_BYTE *)v19;
      v46 = sub_1F58D40((__int64)&v79);
      v21 = v71;
      v22 = v46;
      LOBYTE(v83) = v71;
      v84 = v67;
      if ( v71 )
        goto LABEL_9;
      goto LABEL_37;
    }
    if ( v80 != *(const void ***)(v19 + 8) && !(_BYTE)v79 )
      goto LABEL_36;
  }
LABEL_19:
  if ( a3 - 122 > 2 )
    goto LABEL_26;
  v31 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v31 > 0x17u )
  {
    if ( (unsigned __int8)v31 <= 0x2Fu && (v47 = 0x80A800000000LL, _bittest64(&v47, v31)) )
    {
      v48 = *(_BYTE *)(a2 + 17);
      v32 = (v48 & 4) != 0;
      v33 = (v48 & 2) != 0;
    }
    else
    {
      v32 = 0;
      v33 = 0;
    }
    v34 = (unsigned __int8)(v31 - 48) <= 1u || (unsigned int)(unsigned __int8)v31 - 41 <= 1;
    if ( v34 )
      goto LABEL_24;
    goto LABEL_27;
  }
  if ( (_BYTE)v31 != 5 )
  {
LABEL_26:
    v34 = 0;
    v32 = 0;
    v33 = 0;
    goto LABEL_27;
  }
  v49 = *(unsigned __int16 *)(a2 + 18);
  if ( (unsigned __int16)v49 > 0x17u )
  {
    v33 = 0;
    v32 = 0;
    v51 = (unsigned __int16)v49;
  }
  else
  {
    v50 = &loc_80A800;
    v51 = (unsigned __int16)v49;
    if ( _bittest64((const __int64 *)&v50, v49) )
    {
      v52 = *(_BYTE *)(a2 + 17);
      v32 = (v52 & 4) != 0;
      v33 = (v52 & 2) != 0;
    }
    else
    {
      v33 = 0;
      v32 = 0;
    }
  }
  v34 = (unsigned int)(v51 - 17) <= 1 || (unsigned __int16)(v49 - 24) <= 1u;
  if ( v34 )
LABEL_24:
    v34 = (*(_BYTE *)(a2 + 17) & 2) != 0;
LABEL_27:
  v35 = *(__int64 **)(a1 + 552);
  v36 = (2 * v33) | (4 * v32) | (8 * (unsigned int)v34 + 1);
  v37 = *(_DWORD *)(a1 + 536);
  v38 = *(const void ***)(v73[5] + 16LL * (unsigned int)v74 + 8);
  v39 = *(unsigned __int8 *)(v73[5] + 16LL * (unsigned int)v74);
  v83 = 0;
  v40 = *(_QWORD *)a1;
  LODWORD(v84) = v37;
  if ( v40 )
  {
    if ( &v83 != (__int64 *)(v40 + 48) )
    {
      v41 = *(_QWORD *)(v40 + 48);
      v83 = v41;
      if ( v41 )
      {
        v64 = v39;
        v66 = v36;
        v70 = v38;
        sub_1623A60((__int64)&v83, v41, 2);
        v39 = v64;
        v36 = v66;
        v38 = v70;
      }
    }
  }
  *((_QWORD *)&v63 + 1) = v12 | *((_QWORD *)&v78 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v63 = v76;
  v42 = sub_1D332F0(
          v35,
          a3,
          (__int64)&v83,
          v39,
          v38,
          v36,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          (__int64)v73,
          v74,
          v63);
  v44 = v43;
  if ( v83 )
    sub_161E7C0((__int64)&v83, v83);
  v83 = a2;
  result = sub_205F5C0(a1 + 8, &v83);
  result[1] = (__int64)v42;
  *((_DWORD *)result + 4) = v44;
  return result;
}

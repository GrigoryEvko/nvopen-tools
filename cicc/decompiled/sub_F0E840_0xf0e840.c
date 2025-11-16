// Function: sub_F0E840
// Address: 0xf0e840
//
_QWORD *__fastcall sub_F0E840(
        const __m128i *a1,
        unsigned __int8 *a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        unsigned __int8 *a7)
{
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // edi
  __int64 v19; // rax
  __int64 v20; // rcx
  char v21; // dl
  int v22; // edi
  char v23; // r13
  unsigned int v24; // ecx
  int v25; // eax
  unsigned int v26; // r10d
  char v27; // al
  _QWORD *v28; // r12
  __int64 v30; // r12
  __int64 v31; // rbx
  unsigned __int8 *v32; // r15
  _QWORD *v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  int v36; // ebx
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  __int64 v41; // rax
  bool v42; // al
  int v43; // eax
  _BYTE *v44; // rax
  unsigned __int64 v45; // r8
  char v46; // dl
  bool v47; // zf
  unsigned __int64 v48; // rcx
  __int64 v49; // rsi
  void **v50; // rax
  void **v51; // rcx
  char v52; // al
  int v53; // esi
  void *v54; // rax
  _BYTE *v55; // rcx
  void **v56; // [rsp+0h] [rbp-150h]
  __int64 v57; // [rsp+8h] [rbp-148h]
  unsigned __int8 v58; // [rsp+8h] [rbp-148h]
  char v59; // [rsp+8h] [rbp-148h]
  __int64 v60; // [rsp+10h] [rbp-140h] BYREF
  unsigned __int64 v61; // [rsp+18h] [rbp-138h]
  unsigned __int64 v62; // [rsp+20h] [rbp-130h]
  int v63; // [rsp+28h] [rbp-128h]
  unsigned __int8 v64; // [rsp+2Ch] [rbp-124h] BYREF
  char v65; // [rsp+2Fh] [rbp-121h]
  unsigned int v66; // [rsp+30h] [rbp-120h] BYREF
  int v67; // [rsp+34h] [rbp-11Ch] BYREF
  unsigned int v68; // [rsp+38h] [rbp-118h] BYREF
  unsigned int v69; // [rsp+3Ch] [rbp-114h]
  _QWORD v70[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v71[4]; // [rsp+50h] [rbp-100h] BYREF
  char v72[32]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v73; // [rsp+90h] [rbp-C0h]
  _BYTE v74[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v75; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v76[16]; // [rsp+D0h] [rbp-80h] BYREF

  v60 = a4;
  v10 = *((_QWORD *)a2 + 1);
  v61 = a5;
  v11 = *(_QWORD *)(a4 + 8);
  v64 = a3;
  v12 = v10;
  v66 = sub_BCB060(v11);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v12 = **(_QWORD **)(v10 + 16);
  v16 = sub_BCAC60(v12, (__int64)a2, v13, v14, v15);
  v71[1] = a1;
  v67 = sub_C336A0(v16);
  v76[0] = &v64;
  v68 = v66;
  v69 = v66;
  v76[2] = (unsigned __int8 *)v70;
  v71[0] = a7;
  v76[3] = (unsigned __int8 *)&v67;
  v71[2] = &v60;
  v76[4] = (unsigned __int8 *)&v66;
  v76[6] = (unsigned __int8 *)&v60;
  v70[0] = a7;
  v70[1] = a1;
  v76[1] = a2;
  v76[5] = (unsigned __int8 *)&v68;
  v76[7] = (unsigned __int8 *)a1;
  v76[8] = a7;
  v76[9] = (unsigned __int8 *)v71;
  if ( a6 )
  {
    if ( v64 )
    {
      if ( *a2 == 47 )
      {
        if ( *(_BYTE *)a6 == 18 )
        {
          if ( *(void **)(a6 + 24) == sub_C33340() )
            v41 = *(_QWORD *)(a6 + 32);
          else
            v41 = a6 + 24;
          v42 = (*(_BYTE *)(v41 + 20) & 7) != 3;
        }
        else
        {
          v58 = v64;
          v43 = *(unsigned __int8 *)(*(_QWORD *)(a6 + 8) + 8LL);
          v62 = *(_QWORD *)(a6 + 8);
          if ( (unsigned int)(v43 - 17) > 1 )
            return 0;
          v44 = sub_AD7630(a6, 0, v64);
          v45 = v62;
          v46 = v58;
          if ( !v44 || (v47 = *v44 == 18, v62 = (unsigned __int64)v44, !v47) )
          {
            if ( *(_BYTE *)(v45 + 8) == 17 )
            {
              v63 = *(_DWORD *)(v45 + 32);
              if ( v63 )
              {
                v65 = 0;
                v49 = 0;
                while ( 1 )
                {
                  v59 = v46;
                  LODWORD(v62) = v49;
                  v50 = (void **)sub_AD69F0((unsigned __int8 *)a6, v49);
                  v51 = v50;
                  if ( !v50 )
                    break;
                  v52 = *(_BYTE *)v50;
                  v56 = v51;
                  v53 = v62;
                  v46 = v59;
                  if ( v52 != 13 )
                  {
                    LOBYTE(v62) = v59;
                    if ( v52 != 18 )
                      return 0;
                    v54 = sub_C33340();
                    v46 = v62;
                    v55 = v56[3] == v54 ? v56[4] : v56 + 3;
                    if ( (v55[20] & 7) == 3 )
                      return 0;
                    v65 = v62;
                  }
                  v49 = (unsigned int)(v53 + 1);
                  if ( v63 == (_DWORD)v49 )
                  {
                    if ( v65 )
                      goto LABEL_47;
                    return 0;
                  }
                }
              }
            }
            return 0;
          }
          if ( *(void **)(v62 + 24) == sub_C33340() )
            v48 = *(_QWORD *)(v62 + 32);
          else
            v48 = v62 + 24;
          v42 = (*(_BYTE *)(v48 + 20) & 7) != 3;
        }
        if ( !v42 )
          return 0;
LABEL_47:
        v17 = a1[5].m128i_i64[1];
        v18 = 42 - (v64 == 0);
      }
      else
      {
        v17 = a1[5].m128i_i64[1];
        v18 = 42;
      }
    }
    else
    {
      v17 = a1[5].m128i_i64[1];
      v18 = 41;
    }
    v19 = sub_96F480(v18, a6, v11, v17);
    if ( !v19 )
      return 0;
    v20 = a1[5].m128i_i64[1];
    v62 = v19;
    if ( a6 != sub_96F480(43 - ((unsigned int)(v64 == 0) - 1), v19, v10, v20) )
      return 0;
    v61 = v62;
    if ( v11 != *(_QWORD *)(v62 + 8) )
      return 0;
  }
  else if ( v11 != *(_QWORD *)(v61 + 8) || !(unsigned __int8)sub_F08E70(v76, 1u) )
  {
    return 0;
  }
  v21 = sub_F08E70(v76, 0);
  if ( !v21 )
    return 0;
  v22 = *a2;
  v23 = v64;
  v24 = v69;
  v25 = 1 - ((v64 == 0) - 1);
  if ( v68 >= v69 )
    v24 = v68;
  switch ( v22 )
  {
    case '-':
      v26 = 15;
      if ( v24 + v25 < v66 )
      {
        v23 = v21;
        v26 = 15;
        goto LABEL_22;
      }
      break;
    case '/':
      v26 = 17;
      if ( v25 + 2 * v24 < v66 )
        goto LABEL_22;
      break;
    case '+':
      v26 = 13;
      if ( v24 + v25 < v66 )
        goto LABEL_22;
      break;
    default:
      BUG();
  }
  LODWORD(v62) = v26;
  v27 = sub_F0C210(a1, v26, v60, v61, (__int64)a2, v64);
  v26 = v62;
  if ( !v27 )
    return 0;
LABEL_22:
  v30 = a1[2].m128i_i64[0];
  LODWORD(v62) = v26;
  v73 = 257;
  v31 = v60;
  v57 = v61;
  v32 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, unsigned __int64))(**(_QWORD **)(v30 + 80)
                                                                                                + 16LL))(
                             *(_QWORD *)(v30 + 80),
                             v26,
                             v60,
                             v61);
  if ( !v32 )
  {
    v75 = 257;
    v32 = (unsigned __int8 *)sub_B504D0(v62, v31, v57, (__int64)v74, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v32) )
    {
      v35 = *(_QWORD *)(v30 + 96);
      v36 = *(_DWORD *)(v30 + 104);
      if ( v35 )
        sub_B99FD0((__int64)v32, 3u, v35);
      sub_B45150((__int64)v32, v36);
    }
    (*(void (__fastcall **)(_QWORD, unsigned __int8 *, char *, _QWORD, _QWORD))(**(_QWORD **)(v30 + 88) + 16LL))(
      *(_QWORD *)(v30 + 88),
      v32,
      v72,
      *(_QWORD *)(v30 + 56),
      *(_QWORD *)(v30 + 64));
    v37 = *(_QWORD *)v30;
    v38 = *(_QWORD *)v30 + 16LL * *(unsigned int *)(v30 + 8);
    while ( v38 != v37 )
    {
      v39 = *(_QWORD *)(v37 + 8);
      v40 = *(_DWORD *)v37;
      v37 += 16;
      sub_B99FD0((__int64)v32, v40, v39);
    }
  }
  if ( (unsigned __int8)(*v32 - 42) <= 0x11u )
  {
    sub_B44850(v32, v23);
    sub_B447F0(v32, v23 ^ 1);
  }
  v75 = 257;
  if ( v23 )
  {
    v33 = sub_BD2C40(72, unk_3F10A14);
    v28 = v33;
    if ( v33 )
      sub_B518D0((__int64)v33, (__int64)v32, v10, (__int64)v74, 0, 0);
  }
  else
  {
    v34 = sub_BD2C40(72, unk_3F10A14);
    v28 = v34;
    if ( v34 )
      sub_B51830((__int64)v34, (__int64)v32, v10, (__int64)v74, 0, 0);
  }
  return v28;
}

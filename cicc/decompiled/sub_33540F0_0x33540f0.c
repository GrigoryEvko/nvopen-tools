// Function: sub_33540F0
// Address: 0x33540f0
//
__int64 __fastcall sub_33540F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  unsigned int v10; // r14d
  unsigned int v11; // ebx
  __int64 v12; // r13
  __int64 v13; // r8
  __int64 v14; // r12
  __int64 v15; // r10
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // eax
  __int64 *v22; // r8
  __int64 v23; // r10
  unsigned int v25; // eax
  bool v26; // al
  bool v27; // cf
  __int64 v28; // rax
  char v29; // r11
  __int64 v30; // rdi
  __int64 (*v31)(); // rax
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 (*v34)(); // rax
  int v35; // eax
  unsigned int v36; // r12d
  bool v37; // al
  int v38; // eax
  int v39; // eax
  unsigned int v40; // r12d
  int v41; // eax
  int v42; // eax
  __int64 v43; // [rsp-70h] [rbp-70h]
  __int64 v44; // [rsp-70h] [rbp-70h]
  unsigned __int8 v45; // [rsp-68h] [rbp-68h]
  unsigned __int8 v46; // [rsp-68h] [rbp-68h]
  int v47; // [rsp-68h] [rbp-68h]
  int v48; // [rsp-68h] [rbp-68h]
  unsigned __int8 v49; // [rsp-68h] [rbp-68h]
  int v50; // [rsp-68h] [rbp-68h]
  char v51; // [rsp-68h] [rbp-68h]
  __int64 v52; // [rsp-60h] [rbp-60h]
  __int64 v53; // [rsp-60h] [rbp-60h]
  __int64 v54; // [rsp-60h] [rbp-60h]
  __int64 v55; // [rsp-60h] [rbp-60h]
  __int64 v56; // [rsp-60h] [rbp-60h]
  __int64 v57; // [rsp-60h] [rbp-60h]
  __int64 v58; // [rsp-60h] [rbp-60h]
  __int64 v59; // [rsp-60h] [rbp-60h]
  __int64 v60; // [rsp-60h] [rbp-60h]
  __int64 v61; // [rsp-60h] [rbp-60h]
  __int64 v62; // [rsp-60h] [rbp-60h]
  __int64 v63; // [rsp-60h] [rbp-60h]
  __int64 v64; // [rsp-60h] [rbp-60h]
  __int64 v65; // [rsp-60h] [rbp-60h]
  __int64 v66; // [rsp-60h] [rbp-60h]
  int v67; // [rsp-58h] [rbp-58h]
  __int64 v68; // [rsp-58h] [rbp-58h]
  __int64 v69; // [rsp-58h] [rbp-58h]
  __int64 v70; // [rsp-58h] [rbp-58h]
  __int64 v71; // [rsp-58h] [rbp-58h]
  __int64 v72; // [rsp-58h] [rbp-58h]
  __int64 v73; // [rsp-58h] [rbp-58h]
  __int64 v74; // [rsp-58h] [rbp-58h]
  __int64 v75; // [rsp-58h] [rbp-58h]
  __int64 v76; // [rsp-58h] [rbp-58h]
  __int64 v77; // [rsp-58h] [rbp-58h]
  __int64 v78; // [rsp-58h] [rbp-58h]
  __int64 v79; // [rsp-58h] [rbp-58h]
  __int64 v80; // [rsp-58h] [rbp-58h]
  __int64 v81; // [rsp-58h] [rbp-58h]
  __int64 v82; // [rsp-58h] [rbp-58h]
  char v83; // [rsp-4Dh] [rbp-4Dh]
  int v84; // [rsp-4Ch] [rbp-4Ch]
  unsigned int v85; // [rsp-40h] [rbp-40h] BYREF
  unsigned int v86; // [rsp-3Ch] [rbp-3Ch] BYREF

  v6 = (_QWORD *)a1[3];
  v7 = a1[2];
  if ( v6 == (_QWORD *)v7 )
    return 0;
  v8 = (__int64)v6 - v7;
  if ( (unsigned __int64)v6 - v7 > 0x1F40 )
  {
    v84 = 1000;
  }
  else
  {
    v84 = v8 >> 3;
    if ( v8 == 8 )
    {
      v23 = *(_QWORD *)v7;
      goto LABEL_17;
    }
  }
  v10 = 0;
  v11 = 1;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v7 + 8LL * v11);
      v13 = 8LL * v11;
      v14 = *(_QWORD *)(v7 + 8LL * v10);
      v15 = 8LL * v10;
      v16 = (*(_BYTE *)(v12 + 249) & 0x10) != 0;
      v17 = (*(_BYTE *)(v14 + 249) & 0x10) != 0;
      if ( (_BYTE)v17 != v16 )
      {
        if ( (unsigned __int8)v17 >= v16 )
          goto LABEL_13;
        goto LABEL_49;
      }
      if ( (*(_BYTE *)(v14 + 248) & 2) != 0 || (*(_BYTE *)(v12 + 248) & 2) != 0 )
      {
        v37 = sub_3353760(v14, *(_QWORD *)(v7 + 8LL * v11), a1[21]);
        v7 = a1[2];
        v13 = 8LL * v11;
        v15 = 8LL * v10;
        goto LABEL_53;
      }
      v85 = 0;
      v86 = 0;
      if ( byte_5038E28 && (_BYTE)qword_5038D48 )
        goto LABEL_24;
      v7 = v12;
      v67 = sub_3351EF0((_QWORD *)a1[21], (__int64 *)v14, &v85, a4, v13, a6);
      v21 = sub_3351EF0((_QWORD *)a1[21], (__int64 *)v12, &v86, v18, v19, v20);
      v13 = 8LL * v11;
      v15 = 8LL * v10;
      if ( byte_5038E28 )
        goto LABEL_22;
      if ( v67 == v21 )
        break;
      v7 = a1[2];
      if ( v67 > v21 )
        goto LABEL_49;
LABEL_13:
      v13 = v15;
LABEL_14:
      if ( ++v11 == v84 )
        goto LABEL_15;
    }
    if ( v67 <= 0 )
      goto LABEL_22;
    LOBYTE(v25) = sub_3351690((_DWORD *)v14);
    v7 = v25;
    v26 = sub_3351690((_DWORD *)v12);
    if ( v26 || !(_BYTE)v7 )
    {
      if ( (_BYTE)v7 == 1 || !v26 )
      {
LABEL_22:
        if ( (_BYTE)qword_5038D48 || (v27 = v85 < v86, v85 == v86) )
        {
LABEL_24:
          a4 = (unsigned __int8)byte_5038AA8;
          if ( byte_5038AA8 )
            goto LABEL_81;
          v28 = a1[21];
          if ( (*(_BYTE *)(v14 + 254) & 2) == 0 )
          {
            v43 = a1[21];
            v45 = byte_5038AA8;
            v53 = v15;
            v69 = v13;
            sub_2F8F770(v14, (_QWORD *)v7, v17, (unsigned __int8)byte_5038AA8, v13, a6);
            v28 = v43;
            a4 = v45;
            v15 = v53;
            v13 = v69;
          }
          v29 = 1;
          if ( *(_DWORD *)(v14 + 244) <= *(_DWORD *)(v28 + 8) )
          {
            v29 = 0;
            v30 = *(_QWORD *)(*(_QWORD *)(v28 + 88) + 672LL);
            v31 = *(__int64 (**)())(*(_QWORD *)v30 + 24LL);
            if ( v31 != sub_2EC0B50 )
            {
              v49 = a4;
              v7 = v14;
              v61 = v15;
              v77 = v13;
              v41 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v31)(v30, v14, 0);
              a4 = v49;
              v15 = v61;
              v13 = v77;
              v29 = v41 != 0;
            }
          }
          v32 = a1[21];
          if ( (*(_BYTE *)(v12 + 254) & 2) == 0 )
          {
            v83 = v29;
            v44 = a1[21];
            v46 = a4;
            v54 = v15;
            v70 = v13;
            sub_2F8F770(v12, (_QWORD *)v7, v17, a4, v13, a6);
            v29 = v83;
            v32 = v44;
            a4 = v46;
            v15 = v54;
            v13 = v70;
          }
          if ( *(_DWORD *)(v12 + 244) > *(_DWORD *)(v32 + 8) )
          {
            a4 = 1;
          }
          else
          {
            v33 = *(_QWORD *)(*(_QWORD *)(v32 + 88) + 672LL);
            v34 = *(__int64 (**)())(*(_QWORD *)v33 + 24LL);
            if ( v34 != sub_2EC0B50 )
            {
              v51 = v29;
              v7 = v12;
              v64 = v15;
              v80 = v13;
              v42 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v34)(v33, v12, 0);
              v13 = v80;
              v15 = v64;
              v29 = v51;
              LOBYTE(a4) = v42 != 0;
            }
          }
          if ( (_BYTE)a4 == v29 )
          {
LABEL_81:
            if ( !(_BYTE)qword_50389C8 )
            {
              if ( (*(_BYTE *)(v14 + 254) & 1) == 0 )
              {
                v58 = v15;
                v74 = v13;
                sub_2F8F5D0(v14, (_QWORD *)v7, v17, a4, v13, a6);
                v15 = v58;
                v13 = v74;
              }
              v35 = *(_DWORD *)(v14 + 240);
              if ( (*(_BYTE *)(v12 + 254) & 1) == 0 )
              {
                v47 = *(_DWORD *)(v14 + 240);
                v57 = v15;
                v73 = v13;
                sub_2F8F5D0(v12, (_QWORD *)v7, v17, a4, v13, a6);
                v35 = v47;
                v15 = v57;
                v13 = v73;
              }
              v17 = (unsigned int)((v35 - *(_DWORD *)(v12 + 240)) >> 31);
              if ( (int)((v17 ^ (v35 - *(_DWORD *)(v12 + 240))) - v17) > (int)qword_5038728 )
              {
                if ( (*(_BYTE *)(v14 + 254) & 1) == 0 )
                {
                  v66 = v15;
                  v82 = v13;
                  sub_2F8F5D0(v14, (_QWORD *)v7, (unsigned int)v17, a4, v13, a6);
                  v15 = v66;
                  v13 = v82;
                }
                v36 = *(_DWORD *)(v14 + 240);
                if ( (*(_BYTE *)(v12 + 254) & 1) == 0 )
                {
                  v65 = v15;
                  v81 = v13;
                  sub_2F8F5D0(v12, (_QWORD *)v7, v17, a4, v13, a6);
                  v15 = v65;
                  v13 = v81;
                }
                v27 = v36 < *(_DWORD *)(v12 + 240);
                goto LABEL_47;
              }
            }
            if ( byte_50388E8 )
              goto LABEL_71;
            if ( (*(_BYTE *)(v14 + 254) & 2) == 0 )
            {
              v60 = v15;
              v76 = v13;
              sub_2F8F770(v14, (_QWORD *)v7, v17, a4, v13, a6);
              v15 = v60;
              v13 = v76;
            }
            v38 = *(_DWORD *)(v14 + 244);
            if ( (*(_BYTE *)(v12 + 254) & 2) == 0 )
            {
              v48 = *(_DWORD *)(v14 + 244);
              v59 = v15;
              v75 = v13;
              sub_2F8F770(v12, (_QWORD *)v7, v17, a4, v13, a6);
              v38 = v48;
              v15 = v59;
              v13 = v75;
            }
            if ( v38 == *(_DWORD *)(v12 + 244) )
              goto LABEL_71;
            if ( (*(_BYTE *)(v14 + 254) & 2) == 0 )
            {
              v63 = v15;
              v79 = v13;
              sub_2F8F770(v14, (_QWORD *)v7, v17, a4, v13, a6);
              v15 = v63;
              v13 = v79;
            }
            v39 = *(_DWORD *)(v14 + 244);
            if ( (*(_BYTE *)(v12 + 254) & 2) == 0 )
            {
              v50 = *(_DWORD *)(v14 + 244);
              v62 = v15;
              v78 = v13;
              sub_2F8F770(v12, (_QWORD *)v7, v17, a4, v13, a6);
              v39 = v50;
              v15 = v62;
              v13 = v78;
            }
            v17 = (unsigned int)((v39 - *(_DWORD *)(v12 + 244)) >> 31);
            if ( (int)((v17 ^ (v39 - *(_DWORD *)(v12 + 244))) - v17) <= (int)qword_5038728 )
            {
LABEL_71:
              v52 = v15;
              v68 = v13;
              v37 = sub_3353760(v14, v12, a1[21]);
              v7 = a1[2];
              v15 = v52;
              v13 = v68;
              goto LABEL_53;
            }
          }
          if ( (*(_BYTE *)(v14 + 254) & 2) == 0 )
          {
            v56 = v15;
            v72 = v13;
            sub_2F8F770(v14, (_QWORD *)v7, v17, a4, v13, a6);
            v15 = v56;
            v13 = v72;
          }
          v40 = *(_DWORD *)(v14 + 244);
          if ( (*(_BYTE *)(v12 + 254) & 2) == 0 )
          {
            v55 = v15;
            v71 = v13;
            sub_2F8F770(v12, (_QWORD *)v7, v17, a4, v13, a6);
            v15 = v55;
            v13 = v71;
          }
          v7 = a1[2];
          v37 = v40 > *(_DWORD *)(v12 + 244);
          goto LABEL_53;
        }
LABEL_47:
        v7 = a1[2];
        v37 = v27;
LABEL_53:
        if ( !v37 )
        {
          v13 = v15;
          goto LABEL_14;
        }
LABEL_49:
        v10 = v11;
        goto LABEL_50;
      }
      v7 = a1[2];
      v10 = v11;
    }
    else
    {
      v7 = a1[2];
      v13 = v15;
    }
LABEL_50:
    ++v11;
  }
  while ( v11 != v84 );
LABEL_15:
  v6 = (_QWORD *)a1[3];
  v22 = (__int64 *)(v7 + v13);
  v23 = *v22;
  if ( v10 + 1 != ((__int64)v6 - v7) >> 3 )
  {
    *v22 = *(v6 - 1);
    *(v6 - 1) = v23;
    v6 = (_QWORD *)a1[3];
  }
LABEL_17:
  a1[3] = v6 - 1;
  *(_DWORD *)(v23 + 204) = 0;
  return v23;
}

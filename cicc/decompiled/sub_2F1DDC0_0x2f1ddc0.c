// Function: sub_2F1DDC0
// Address: 0x2f1ddc0
//
__int64 __fastcall sub_2F1DDC0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // r15
  char i; // al
  __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  char v10; // al
  __int64 v11; // rcx
  unsigned __int64 *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r14
  const char *v15; // rsi
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // r13
  int v19; // ebx
  __int64 v20; // r13
  __int64 v21; // r15
  unsigned __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // r13
  unsigned __int64 *v29; // rbx
  unsigned __int64 *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  const char *v33; // rax
  __int64 v34; // rdx
  void (__fastcall *v35)(__int64, char **); // rcx
  unsigned __int64 *v36; // rbx
  unsigned __int64 *v37; // r12
  size_t v38; // rdx
  __int64 v39; // rax
  char *v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // [rsp+10h] [rbp-1E0h]
  unsigned __int16 *v43; // [rsp+18h] [rbp-1D8h]
  __int64 v44; // [rsp+18h] [rbp-1D8h]
  unsigned __int64 v45; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 *v46; // [rsp+38h] [rbp-1B8h]
  __int64 v48; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v49; // [rsp+50h] [rbp-1A0h]
  __int64 v50; // [rsp+60h] [rbp-190h]
  __int64 v51; // [rsp+70h] [rbp-180h]
  __int64 v52; // [rsp+70h] [rbp-180h]
  unsigned __int64 v53; // [rsp+78h] [rbp-178h]
  char v54; // [rsp+8Eh] [rbp-162h] BYREF
  char v55; // [rsp+8Fh] [rbp-161h] BYREF
  __int64 v56; // [rsp+90h] [rbp-160h] BYREF
  const char *v57; // [rsp+98h] [rbp-158h] BYREF
  __int64 v58; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-148h] BYREF
  _QWORD v60[2]; // [rsp+B0h] [rbp-140h] BYREF
  unsigned __int64 *v61; // [rsp+C0h] [rbp-130h] BYREF
  unsigned __int64 *v62; // [rsp+C8h] [rbp-128h]
  __int64 v63; // [rsp+D0h] [rbp-120h]
  void *v64; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v65; // [rsp+E8h] [rbp-108h]
  __int64 v66; // [rsp+F0h] [rbp-100h]
  __int64 v67; // [rsp+F8h] [rbp-F8h]
  __int64 v68; // [rsp+100h] [rbp-F0h]
  __int64 v69; // [rsp+108h] [rbp-E8h]
  char **v70; // [rsp+110h] [rbp-E0h]
  char *v71; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+128h] [rbp-C8h]
  __int64 v73; // [rsp+130h] [rbp-C0h]
  char v74; // [rsp+138h] [rbp-B8h] BYREF
  __int16 v75; // [rsp+140h] [rbp-B0h]

  v2 = a1;
  LODWORD(v3) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = (a2[1] - *a2) >> 5;
  if ( (_DWORD)v3 )
  {
    v4 = 0;
    v5 = a1;
    v53 = 1;
    v50 = (unsigned int)(v3 - 1) + 2LL;
    for ( i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)v5 + 32LL))(v5, 0, &v56);
          ;
          i = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)v5 + 32LL))(v5, (unsigned int)v4, &v56) )
    {
      if ( i )
      {
        v7 = *a2;
        v8 = a2[1];
        v9 = (v8 - *a2) >> 5;
        if ( v9 <= v4 )
        {
          if ( v9 < v53 )
          {
            sub_2F1DB20((__int64)a2, v53 - v9);
            v7 = *a2;
          }
          else if ( v9 > v53 )
          {
            v27 = a2[1];
            v52 = v7 + 32 * v53;
            if ( v8 != v52 )
            {
              v49 = v4;
              v28 = v7 + 32 * v53;
              do
              {
                v29 = *(unsigned __int64 **)(v28 + 16);
                v30 = *(unsigned __int64 **)(v28 + 8);
                if ( v29 != v30 )
                {
                  do
                  {
                    if ( (unsigned __int64 *)*v30 != v30 + 2 )
                      j_j___libc_free_0(*v30);
                    v30 += 7;
                  }
                  while ( v29 != v30 );
                  v30 = *(unsigned __int64 **)(v28 + 8);
                }
                if ( v30 )
                  j_j___libc_free_0((unsigned __int64)v30);
                v28 += 32;
              }
              while ( v27 != v28 );
              v4 = v49;
              a2[1] = v52;
              v7 = *a2;
            }
          }
        }
        v51 = v7 + 32 * v4;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 144LL))(v5);
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void **, char **))(*(_QWORD *)v5 + 120LL))(
               v5,
               "bb",
               1,
               0,
               &v64,
               &v71) )
        {
          sub_2F07DB0(v5, (unsigned int *)v51);
          (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v5 + 128LL))(v5, v71);
        }
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void **, char **))(*(_QWORD *)v5 + 120LL))(
               v5,
               "offset",
               1,
               0,
               &v64,
               &v71) )
        {
          sub_2F07DB0(v5, (unsigned int *)(v51 + 4));
          (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v5 + 128LL))(v5, v71);
        }
        v61 = 0;
        v62 = 0;
        v63 = 0;
        v46 = (unsigned __int64 *)(v51 + 8);
        v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 16LL))(v5);
        v11 = 0;
        if ( v10 )
        {
          v12 = v61;
          v13 = *(_QWORD *)(v51 + 8);
          v14 = *(_QWORD *)(v51 + 16);
          if ( v14 - v13 == (char *)v62 - (char *)v61 )
          {
            if ( v13 == v14 )
            {
LABEL_73:
              v11 = 1;
            }
            else
            {
              while ( 1 )
              {
                v38 = *(_QWORD *)(v13 + 8);
                if ( v38 != v12[1] || v38 && memcmp(*(const void **)v13, (const void *)*v12, v38) )
                  break;
                if ( *(_WORD *)(v13 + 48) != *((_WORD *)v12 + 24) )
                  break;
                v13 += 56;
                v12 += 7;
                if ( v14 == v13 )
                  goto LABEL_73;
              }
              v11 = 0;
            }
          }
        }
        v15 = "fwdArgRegs";
        if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, char *, const char **))(*(_QWORD *)v5 + 120LL))(
               v5,
               "fwdArgRegs",
               0,
               v11,
               &v54,
               &v57) )
        {
          v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 24LL))(v5);
          if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v5 + 16LL))(v5) )
            v19 = -1227133513 * ((__int64)(*(_QWORD *)(v51 + 16) - *(_QWORD *)(v51 + 8)) >> 3);
          if ( v19 )
          {
            v20 = v5;
            v21 = 0;
            v48 = (unsigned int)(v19 - 1) + 2LL;
            v22 = 1;
            v45 = v4;
            do
            {
              while ( 1 )
              {
                v23 = v21 + 56;
                if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)v20 + 32LL))(
                       v20,
                       (unsigned int)(v22 - 1),
                       &v58) )
                {
                  break;
                }
                v21 += 56;
                if ( v48 == ++v22 )
                  goto LABEL_39;
              }
              v24 = *(_QWORD *)(v51 + 8);
              v25 = 0x6DB6DB6DB6DB6DB7LL * ((*(_QWORD *)(v51 + 16) - v24) >> 3);
              if ( v25 <= v22 - 1 )
              {
                if ( v25 < v22 )
                {
                  sub_2F1AA90(v46, v22 - v25);
                  v24 = *(_QWORD *)(v51 + 8);
                }
                else if ( v25 > v22 )
                {
                  v44 = v24 + v23;
                  if ( *(_QWORD *)(v51 + 16) != v24 + v23 )
                  {
                    v42 = v22;
                    v36 = *(unsigned __int64 **)(v51 + 16);
                    v37 = (unsigned __int64 *)(v24 + v23);
                    do
                    {
                      if ( (unsigned __int64 *)*v37 != v37 + 2 )
                        j_j___libc_free_0(*v37);
                      v37 += 7;
                    }
                    while ( v36 != v37 );
                    v22 = v42;
                    v23 = v21 + 56;
                    *(_QWORD *)(v51 + 16) = v44;
                    v24 = *(_QWORD *)(v51 + 8);
                  }
                }
              }
              v26 = v24 + v21;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 144LL))(v20);
              if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)v20 + 120LL))(
                     v20,
                     "arg",
                     1,
                     0,
                     &v55,
                     &v59) )
              {
                v43 = (unsigned __int16 *)(v26 + 48);
                if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 16LL))(v20) )
                {
                  v69 = 0x100000000LL;
                  v71 = &v74;
                  v64 = &unk_49DD288;
                  v72 = 0;
                  v70 = &v71;
                  v73 = 128;
                  v65 = 2;
                  v66 = 0;
                  v67 = 0;
                  v68 = 0;
                  sub_CB5980((__int64)&v64, 0, 0, 0);
                  v39 = sub_CB0A70(v20);
                  sub_CB2CE0(v43, v39, (__int64)&v64);
                  v40 = v70[1];
                  v60[0] = *v70;
                  v41 = *(_QWORD *)v20;
                  v60[1] = v40;
                  (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(v41 + 216))(v20, v60, 0);
                  v64 = &unk_49DD388;
                  sub_CB5840((__int64)&v64);
                  if ( v71 != &v74 )
                    _libc_free((unsigned __int64)v71);
                }
                else
                {
                  v31 = *(_QWORD *)v20;
                  v64 = 0;
                  v65 = 0;
                  (*(void (__fastcall **)(__int64, void **, _QWORD))(v31 + 216))(v20, &v64, 0);
                  v32 = sub_CB0A70(v20);
                  v33 = sub_CB2CF0((__int64)v64, v65, v32, v43);
                  if ( v34 )
                  {
                    v35 = *(void (__fastcall **)(__int64, char **))(*(_QWORD *)v20 + 248LL);
                    v75 = 261;
                    v71 = (char *)v33;
                    v72 = v34;
                    v35(v20, &v71);
                  }
                }
                (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 128LL))(v20, v59);
              }
              if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void **, char **))(*(_QWORD *)v20 + 120LL))(
                     v20,
                     "reg",
                     1,
                     0,
                     &v64,
                     &v71) )
              {
                sub_2F0E9C0(v20, v26);
                (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v20 + 128LL))(v20, v71);
              }
              v21 = v23;
              ++v22;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 152LL))(v20);
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 40LL))(v20, v58);
            }
            while ( v48 != v22 );
LABEL_39:
            v4 = v45;
            v5 = v20;
          }
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL))(v5);
          v15 = v57;
          (*(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v5 + 128LL))(v5, v57);
        }
        else if ( v54 )
        {
          v15 = (const char *)&v61;
          sub_2F092C0((__int64)v46, &v61);
        }
        v16 = v62;
        v17 = v61;
        if ( v62 != v61 )
        {
          do
          {
            if ( (unsigned __int64 *)*v17 != v17 + 2 )
            {
              v15 = (const char *)(v17[2] + 1);
              j_j___libc_free_0(*v17);
            }
            v17 += 7;
          }
          while ( v16 != v17 );
          v17 = v61;
        }
        if ( v17 )
        {
          v15 = (const char *)(v63 - (_QWORD)v17);
          j_j___libc_free_0((unsigned __int64)v17);
        }
        (*(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v5 + 152LL))(v5, v15);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 40LL))(v5, v56);
      }
      ++v53;
      ++v4;
      if ( v50 == v53 )
        break;
    }
    v2 = v5;
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
}

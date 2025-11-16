// Function: sub_397C610
// Address: 0x397c610
//
_QWORD *__fastcall sub_397C610(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  unsigned __int64 v6; // r15
  unsigned __int16 v8; // r12
  __int64 v9; // r15
  _QWORD *result; // rax
  __int64 v11; // rdi
  void (*v12)(); // rax
  void (*v13)(); // r14
  char *v14; // rdx
  void (*v15)(); // r12
  const char *v16; // rax
  char *v17; // rdx
  __int64 *i; // rbx
  __int64 v19; // rbx
  __int64 v20; // r11
  void (*v21)(); // r9
  size_t v22; // r8
  void (*v23)(); // r9
  __int64 v24; // r11
  const char *v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r12
  void (*v28)(); // r13
  const char *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // [rsp+10h] [rbp-310h]
  __int64 v34; // [rsp+10h] [rbp-310h]
  void (*v35)(); // [rsp+18h] [rbp-308h]
  __int64 v36; // [rsp+18h] [rbp-308h]
  void (*v37)(); // [rsp+18h] [rbp-308h]
  __int64 v38; // [rsp+50h] [rbp-2D0h]
  __int64 v39; // [rsp+58h] [rbp-2C8h]
  __int64 v40; // [rsp+58h] [rbp-2C8h]
  __int64 v41; // [rsp+60h] [rbp-2C0h] BYREF
  __int64 v42; // [rsp+68h] [rbp-2B8h] BYREF
  _QWORD v43[2]; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v44; // [rsp+80h] [rbp-2A0h]
  _QWORD v45[2]; // [rsp+A0h] [rbp-280h] BYREF
  __int16 v46; // [rsp+B0h] [rbp-270h]
  _QWORD v47[2]; // [rsp+C0h] [rbp-260h] BYREF
  __int16 v48; // [rsp+D0h] [rbp-250h]
  _QWORD v49[2]; // [rsp+E0h] [rbp-240h] BYREF
  __int16 v50; // [rsp+F0h] [rbp-230h]
  _QWORD v51[2]; // [rsp+100h] [rbp-220h] BYREF
  __int16 v52; // [rsp+110h] [rbp-210h]
  _QWORD v53[2]; // [rsp+120h] [rbp-200h] BYREF
  __int16 v54; // [rsp+130h] [rbp-1F0h]
  const char *v55; // [rsp+140h] [rbp-1E0h] BYREF
  char *v56; // [rsp+148h] [rbp-1D8h]
  _WORD v57[8]; // [rsp+150h] [rbp-1D0h] BYREF
  unsigned __int64 *v58; // [rsp+160h] [rbp-1C0h] BYREF
  _QWORD *v59; // [rsp+168h] [rbp-1B8h]
  __int64 (__fastcall **v60)(); // [rsp+170h] [rbp-1B0h] BYREF
  _QWORD v61[3]; // [rsp+178h] [rbp-1A8h] BYREF
  unsigned __int64 v62; // [rsp+190h] [rbp-190h]
  _BYTE *v63; // [rsp+198h] [rbp-188h]
  unsigned __int64 v64; // [rsp+1A0h] [rbp-180h]
  __int64 v65; // [rsp+1A8h] [rbp-178h]
  volatile signed __int32 *v66; // [rsp+1B0h] [rbp-170h] BYREF
  int v67; // [rsp+1B8h] [rbp-168h]
  unsigned __int64 v68[2]; // [rsp+1C0h] [rbp-160h] BYREF
  _BYTE v69[16]; // [rsp+1D0h] [rbp-150h] BYREF
  _QWORD v70[28]; // [rsp+1E0h] [rbp-140h] BYREF
  __int16 v71; // [rsp+2C0h] [rbp-60h]
  __int64 v72; // [rsp+2C8h] [rbp-58h]
  __int64 v73; // [rsp+2D0h] [rbp-50h]
  __int64 v74; // [rsp+2D8h] [rbp-48h]
  __int64 v75; // [rsp+2E0h] [rbp-40h]

  v2 = a1;
  v3 = a2;
  if ( *(_BYTE *)(a1 + 416) )
  {
    v27 = *(_QWORD *)(a1 + 256);
    v28 = *(void (**)())(*(_QWORD *)v27 + 104LL);
    v29 = sub_14E0540(*(unsigned __int16 *)(a2 + 28));
    v4 = *(unsigned int *)(a2 + 24);
    v43[0] = v29;
    v30 = *(unsigned int *)(v3 + 20);
    LODWORD(v44) = v4;
    v42 = v30;
    v31 = *(unsigned int *)(v3 + 16);
    v43[1] = v32;
    v41 = v31;
    v45[0] = "Abbrev [";
    v46 = 2307;
    v45[1] = v44;
    v47[0] = v45;
    v47[1] = "] 0x";
    v49[0] = v47;
    v49[1] = &v41;
    v51[0] = v49;
    v51[1] = ":0x";
    v53[0] = v51;
    v53[1] = &v42;
    v55 = (const char *)v53;
    v56 = " ";
    v58 = (unsigned __int64 *)&v55;
    v59 = v43;
    v48 = 770;
    v50 = 3842;
    v52 = 770;
    v54 = 3842;
    v57[0] = 770;
    LOWORD(v60) = 1282;
    if ( v28 == nullsub_580 )
      goto LABEL_3;
    ((void (__fastcall *)(__int64, unsigned __int64 **, __int64))v28)(v27, &v58, 1);
  }
  v4 = *(unsigned int *)(v3 + 24);
LABEL_3:
  sub_397C0C0(a1, v4, 0);
  v5 = *(_QWORD **)(v3 + 8);
  if ( v5 )
  {
    v6 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v6 )
    {
      v38 = v3;
      while ( 1 )
      {
        v8 = *(_WORD *)(v6 + 12);
        if ( !*(_BYTE *)(a1 + 416) )
          goto LABEL_28;
        v39 = *(_QWORD *)(a1 + 256);
        v13 = *(void (**)())(*(_QWORD *)v39 + 104LL);
        v55 = sub_14E2A80(v8);
        v56 = v14;
        LOWORD(v60) = 261;
        v58 = (unsigned __int64 *)&v55;
        if ( v13 != nullsub_580 )
          ((void (__fastcall *)(__int64, unsigned __int64 **, __int64))v13)(v39, &v58, 1);
        if ( v8 == 50 )
        {
          v40 = *(_QWORD *)(a1 + 256);
          v15 = *(void (**)())(*(_QWORD *)v40 + 104LL);
          v16 = sub_14E7680(*(_DWORD *)(v6 + 16));
          LOWORD(v60) = 261;
          v55 = v16;
          v56 = v17;
          v58 = (unsigned __int64 *)&v55;
          if ( v15 != nullsub_580 )
            ((void (__fastcall *)(__int64, unsigned __int64 **, __int64))v15)(v40, &v58, 1);
        }
        else
        {
LABEL_28:
          if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 232) + 504LL) - 34) <= 1
            && v8 == 2
            && *(_DWORD *)(v6 + 8) == 1 )
          {
            sub_222DF20((__int64)v70);
            v71 = 0;
            v70[27] = 0;
            v70[0] = off_4A06798;
            v72 = 0;
            v73 = 0;
            v74 = 0;
            v58 = qword_4A072D8;
            v75 = 0;
            *(unsigned __int64 **)((char *)&v58 + qword_4A072D8[-3]) = (unsigned __int64 *)&unk_4A07300;
            v59 = 0;
            sub_222DD70((__int64)&v58 + *(v58 - 3), 0);
            v60 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v61[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            sub_222DD70((__int64)&v61[-1] + (_QWORD)*(v60 - 3), 0);
            v58 = qword_4A07328;
            *(unsigned __int64 **)((char *)&v58 + qword_4A07328[-3]) = (unsigned __int64 *)&unk_4A07378;
            v58 = (unsigned __int64 *)off_4A073F0;
            v70[0] = off_4A07440;
            v60 = off_4A07418;
            v61[0] = off_4A07480;
            v61[1] = 0;
            v61[2] = 0;
            v62 = 0;
            v63 = 0;
            v64 = 0;
            v65 = 0;
            sub_220A990(&v66);
            v67 = 24;
            v68[1] = 0;
            v61[0] = off_4A07080;
            v68[0] = (unsigned __int64)v69;
            v69[0] = 0;
            sub_222DD70((__int64)v70, (__int64)v61);
            sub_223E0D0((__int64 *)&v60, "debug_loc offset ", 17);
            sub_223E760((__int64 *)&v60, *(_QWORD *)(v6 + 16));
            v20 = *(_QWORD *)(a1 + 256);
            v21 = *(void (**)())(*(_QWORD *)v20 + 104LL);
            v55 = (const char *)v57;
            v56 = 0;
            LOBYTE(v57[0]) = 0;
            if ( v64 )
            {
              v33 = v20;
              v35 = v21;
              if ( v64 > v62 )
                v22 = v64 - (_QWORD)v63;
              else
                v22 = v62 - (_QWORD)v63;
              sub_2241130((unsigned __int64 *)&v55, 0, 0, v63, v22);
              v23 = v35;
              v24 = v33;
            }
            else
            {
              v34 = v20;
              v37 = v21;
              sub_2240AE0((unsigned __int64 *)&v55, v68);
              v24 = v34;
              v23 = v37;
            }
            v25 = v55;
            v54 = 257;
            if ( *v55 )
            {
              v53[0] = v55;
              LOBYTE(v54) = 3;
            }
            if ( v23 != nullsub_580 )
            {
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v24, v53, 1);
              v25 = v55;
            }
            if ( v25 != (const char *)v57 )
              j_j___libc_free_0((unsigned __int64)v25);
            v36 = *(_QWORD *)(v6 + 16);
            v26 = sub_396DD80(a1);
            sub_396F390(a1, *(_QWORD *)(*(_QWORD *)(v26 + 144) + 8LL), v36, 4u, 0);
            v58 = (unsigned __int64 *)off_4A073F0;
            v70[0] = off_4A07440;
            v60 = off_4A07418;
            v61[0] = off_4A07080;
            if ( (_BYTE *)v68[0] != v69 )
              j_j___libc_free_0(v68[0]);
            v61[0] = off_4A07480;
            sub_2209150(&v66);
            v58 = qword_4A07328;
            *(unsigned __int64 **)((char *)&v58 + qword_4A07328[-3]) = (unsigned __int64 *)&unk_4A07378;
            v60 = (__int64 (__fastcall **)())qword_4A07288;
            *(_QWORD *)((char *)&v61[-1] + qword_4A07288[-3]) = &unk_4A072B0;
            v58 = qword_4A072D8;
            *(unsigned __int64 **)((char *)&v58 + qword_4A072D8[-3]) = (unsigned __int64 *)&unk_4A07300;
            v59 = 0;
            v70[0] = off_4A06798;
            sub_222E050((__int64)v70);
            goto LABEL_10;
          }
        }
        sub_3982B10(v6 + 8, a1);
LABEL_10:
        v9 = *(_QWORD *)v6;
        if ( (v9 & 4) == 0 )
        {
          v6 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v6 )
            continue;
        }
        v2 = a1;
        v3 = v38;
        break;
      }
    }
  }
  result = *(_QWORD **)(v3 + 32);
  if ( !*(_BYTE *)(v3 + 30) )
  {
    if ( !result )
      return result;
    goto LABEL_24;
  }
  if ( result )
  {
LABEL_24:
    for ( i = (__int64 *)(*result & 0xFFFFFFFFFFFFFFF8LL); i; i = (__int64 *)(v19 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_397C610(v2, i);
      v19 = *i;
      if ( (v19 & 4) != 0 )
        break;
    }
  }
  v11 = *(_QWORD *)(v2 + 256);
  v12 = *(void (**)())(*(_QWORD *)v11 + 104LL);
  v58 = (unsigned __int64 *)"End Of Children Mark";
  LOWORD(v60) = 259;
  if ( v12 != nullsub_580 )
    ((void (__fastcall *)(__int64, unsigned __int64 **, __int64))v12)(v11, &v58, 1);
  return (_QWORD *)sub_396F300(v2, 0);
}

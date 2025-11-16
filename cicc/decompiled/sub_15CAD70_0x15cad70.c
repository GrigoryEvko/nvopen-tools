// Function: sub_15CAD70
// Address: 0x15cad70
//
__int64 __fastcall sub_15CAD70(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  const char *v8; // rdi
  size_t v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int8 v13; // dl
  __int64 v14; // rbx
  __int64 result; // rax
  int v16; // ebx
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  __int64 v19; // r12
  int v20; // r13d
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // r13
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  void (__fastcall *v31)(__int64 *, _QWORD **); // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  void (__fastcall *v37)(__int64 *, _QWORD **); // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r13
  _QWORD *v40; // rbx
  _QWORD *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v44; // rbx
  __int64 v45; // rax
  void (__fastcall *v46)(__int64 *, char **, _QWORD); // [rsp+0h] [rbp-120h]
  unsigned __int64 v47; // [rsp+0h] [rbp-120h]
  _QWORD *v48; // [rsp+8h] [rbp-118h]
  void (__fastcall *v49)(__int64 *, char **, _QWORD); // [rsp+8h] [rbp-118h]
  int v50; // [rsp+8h] [rbp-118h]
  __int64 v51; // [rsp+20h] [rbp-100h]
  unsigned __int8 v52; // [rsp+28h] [rbp-F8h]
  __int64 v53; // [rsp+28h] [rbp-F8h]
  char v54; // [rsp+36h] [rbp-EAh] BYREF
  char v55; // [rsp+37h] [rbp-E9h] BYREF
  __int64 v56; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v57; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+48h] [rbp-D8h] BYREF
  _QWORD v59[2]; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD v60[2]; // [rsp+60h] [rbp-C0h] BYREF
  char *v61; // [rsp+70h] [rbp-B0h] BYREF
  char *v62; // [rsp+78h] [rbp-A8h]
  __m128i v63; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v64; // [rsp+90h] [rbp-90h]
  _QWORD *v65; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-78h]
  _QWORD v67[2]; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD *v68; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v69; // [rsp+C8h] [rbp-58h]
  __int64 v70; // [rsp+D0h] [rbp-50h]
  __int64 v71; // [rsp+D8h] [rbp-48h]
  int v72; // [rsp+E0h] [rbp-40h]
  char **v73; // [rsp+E8h] [rbp-38h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, _QWORD))(*a1 + 96))(
          a1,
          "!Passed",
          7,
          (*(_DWORD *)(*a2 + 8) == 8) | (unsigned __int8)(*(_DWORD *)(*a2 + 8) == 14))
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, _QWORD))(*a1 + 96))(
          a1,
          "!Missed",
          7,
          (*(_DWORD *)(*a2 + 8) == 15) | (unsigned __int8)(*(_DWORD *)(*a2 + 8) == 9))
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, _QWORD))(*a1 + 96))(
          a1,
          "!Analysis",
          9,
          (*(_DWORD *)(*a2 + 8) == 16) | (unsigned __int8)(*(_DWORD *)(*a2 + 8) == 10))
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!AnalysisFPCommute",
          18,
          *(_DWORD *)(*a2 + 8) == 11)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!AnalysisAliasing",
          17,
          *(_DWORD *)(*a2 + 8) == 12) )
  {
    (*(void (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
      a1,
      "!Failure",
      8,
      *(_DWORD *)(*a2 + 8) == 13);
  }
  v3 = *a2;
  v4 = *(_QWORD *)(*a2 + 40);
  v63 = _mm_loadu_si128((const __m128i *)(*a2 + 24));
  v64 = v4;
  v5 = (_BYTE *)sub_1649960(*(_QWORD *)(v3 + 16));
  if ( v6 && *v5 == 1 )
  {
    --v6;
    ++v5;
  }
  v59[0] = v5;
  v7 = *a2;
  v59[1] = v6;
  v8 = *(const char **)(v7 + 48);
  v9 = 0;
  v60[0] = v8;
  if ( v8 )
    v9 = strlen(v8);
  v60[1] = v9;
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, _QWORD **, _QWORD **))(*a1 + 120))(
         a1,
         "Pass",
         1,
         0,
         &v65,
         &v68) )
  {
    sub_15C8D20(a1, (__int64)v60);
    (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 128))(a1, v68);
  }
  v10 = *a2;
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, _QWORD **, _QWORD **))(*a1 + 120))(
         a1,
         "Name",
         1,
         0,
         &v65,
         &v68) )
  {
    sub_15C8D20(a1, v10 + 56);
    (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 128))(a1, v68);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) || v63.m128i_i64[1])
    && (*(unsigned __int8 (__fastcall **)(__int64 *, const char *, _QWORD, _QWORD, _QWORD **, _QWORD **))(*a1 + 120))(
         a1,
         "DebugLoc",
         0,
         0,
         &v65,
         &v68) )
  {
    sub_15C8EB0(a1, v63.m128i_i64);
    (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 128))(a1, v68);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, _QWORD **, _QWORD **))(*a1 + 120))(
         a1,
         "Function",
         1,
         0,
         &v65,
         &v68) )
  {
    sub_15C8D20(a1, (__int64)v59);
    (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 128))(a1, v68);
  }
  v11 = *a1;
  v12 = *a2;
  LOBYTE(v57) = 1;
  v13 = (*(__int64 (__fastcall **)(__int64 *))(v11 + 16))(a1);
  if ( v13 )
    v13 = *(_BYTE *)(v12 + 80) ^ 1;
  v52 = v13;
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
  {
    if ( !*(_BYTE *)(v12 + 80) )
      goto LABEL_28;
  }
  else if ( !*(_BYTE *)(v12 + 80) )
  {
    *(_QWORD *)(v12 + 72) = 0;
    *(_BYTE *)(v12 + 80) = 1;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, const char *, _QWORD, _QWORD, __int64 *, __int64 *))(*a1 + 120))(
         a1,
         "Hotness",
         0,
         v52,
         &v57,
         &v58) )
  {
    v26 = v12 + 72;
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
    {
      LOBYTE(v67[0]) = 0;
      v65 = v67;
      v66 = 0;
      v72 = 1;
      v71 = 0;
      v70 = 0;
      v69 = 0;
      v68 = &unk_49EFBE0;
      v73 = (char **)&v65;
      v45 = sub_16E4080(a1);
      sub_16E5A40(v26, v45, &v68);
      if ( v71 != v69 )
        sub_16E7BA0(&v68);
      v61 = *v73;
      v62 = v73[1];
      (*(void (__fastcall **)(__int64 *, char **, _QWORD))(*a1 + 216))(a1, &v61, 0);
      sub_16E7BC0(&v68);
      if ( v65 != v67 )
        j_j___libc_free_0(v65, v67[0] + 1LL);
    }
    else
    {
      v27 = *a1;
      v61 = 0;
      v62 = 0;
      (*(void (__fastcall **)(__int64 *, char **, _QWORD))(v27 + 216))(a1, &v61, 0);
      v28 = sub_16E4080(a1);
      v29 = sub_16E5A50(v61, v62, v28, v26);
      v66 = v30;
      v65 = (_QWORD *)v29;
      if ( v30 )
      {
        v31 = *(void (__fastcall **)(__int64 *, _QWORD **))(*a1 + 232);
        LOWORD(v70) = 261;
        v68 = &v65;
        v31(a1, &v68);
      }
    }
    (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v58);
  }
  else if ( (_BYTE)v57 && *(_BYTE *)(v12 + 80) )
  {
    *(_BYTE *)(v12 + 80) = 0;
  }
LABEL_28:
  v14 = *a2;
  v53 = *a2;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 56))(a1)
    || (result = 11LL * *(unsigned int *)(v14 + 96)) != 0 )
  {
    result = (*(__int64 (__fastcall **)(__int64 *, char *, _QWORD, _QWORD, char *, __int64 *))(*a1 + 120))(
               a1,
               "Args",
               0,
               0,
               &v54,
               &v56);
    if ( (_BYTE)result )
    {
      v16 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
      if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
        v16 = *(_DWORD *)(v53 + 96);
      if ( !v16 )
      {
LABEL_47:
        (*(void (__fastcall **)(__int64 *))(*a1 + 48))(a1);
        return (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v56);
      }
      v17 = (unsigned int)(v16 - 1);
      v18 = 1;
      v51 = v17 + 2;
      v19 = 88;
      while ( 1 )
      {
        v20 = v18;
        if ( (*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, __int64 *))(*a1 + 32))(
               a1,
               (unsigned int)(v18 - 1),
               &v57) )
        {
          break;
        }
LABEL_40:
        ++v18;
        v19 += 88;
        if ( v51 == v18 )
          goto LABEL_47;
      }
      v21 = *(unsigned int *)(v53 + 96);
      if ( v21 > v18 - 1 )
      {
        v22 = *(_QWORD *)(v53 + 88);
LABEL_44:
        v23 = (_QWORD *)(v22 + v19 - 88);
        (*(void (__fastcall **)(__int64 *))(*a1 + 104))(a1);
        if ( (*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, __int64, _QWORD, char *, __int64 *))(*a1 + 120))(
               a1,
               *v23,
               1,
               0,
               &v55,
               &v58) )
        {
          v48 = v23 + 4;
          if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
          {
            LOBYTE(v67[0]) = 0;
            v65 = v67;
            v66 = 0;
            v73 = (char **)&v65;
            v72 = 1;
            v71 = 0;
            v70 = 0;
            v69 = 0;
            v68 = &unk_49EFBE0;
            v24 = sub_16E4080(a1);
            sub_16E5830(v48, v24, &v68);
            if ( v71 != v69 )
              sub_16E7BA0(&v68);
            v61 = *v73;
            v62 = v73[1];
            v49 = *(void (__fastcall **)(__int64 *, char **, _QWORD))(*a1 + 216);
            v25 = sub_15C8A80(v61, (unsigned __int64)v62);
            v49(a1, &v61, v25);
            sub_16E7BC0(&v68);
            if ( v65 != v67 )
              j_j___libc_free_0(v65, v67[0] + 1LL);
          }
          else
          {
            v32 = *a1;
            v61 = 0;
            v62 = 0;
            v46 = *(void (__fastcall **)(__int64 *, char **, _QWORD))(v32 + 216);
            v33 = sub_15C8A80(0, 0);
            v46(a1, &v61, v33);
            v34 = sub_16E4080(a1);
            v35 = sub_16E5850(v61, v62, v34, v48);
            v66 = v36;
            v65 = (_QWORD *)v35;
            if ( v36 )
            {
              v37 = *(void (__fastcall **)(__int64 *, _QWORD **))(*a1 + 232);
              v68 = &v65;
              LOWORD(v70) = 261;
              v37(a1, &v68);
            }
          }
          (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v58);
        }
        if ( v23[9] )
        {
          if ( (*(unsigned __int8 (__fastcall **)(__int64 *, const char *, _QWORD, _QWORD, _QWORD **, _QWORD **))(*a1 + 120))(
                 a1,
                 "DebugLoc",
                 0,
                 0,
                 &v65,
                 &v68) )
          {
            sub_15C8EB0(a1, v23 + 8);
            (*(void (__fastcall **)(__int64 *, _QWORD *))(*a1 + 128))(a1, v68);
          }
        }
        (*(void (__fastcall **)(__int64 *))(*a1 + 112))(a1);
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 40))(a1, v57);
        goto LABEL_40;
      }
      if ( v21 > v18 )
      {
        v22 = *(_QWORD *)(v53 + 88);
        v38 = (_QWORD *)(v22 + 88 * v21);
        if ( v38 != (_QWORD *)(v22 + v19) )
        {
          v50 = v18;
          v39 = (_QWORD *)(v22 + v19);
          v47 = v18;
          v40 = v38;
          do
          {
            v40 -= 11;
            v41 = (_QWORD *)v40[4];
            if ( v41 != v40 + 6 )
              j_j___libc_free_0(v41, v40[6] + 1LL);
            if ( (_QWORD *)*v40 != v40 + 2 )
              j_j___libc_free_0(*v40, v40[2] + 1LL);
          }
          while ( v39 != v40 );
          goto LABEL_72;
        }
      }
      else
      {
        if ( v21 >= v18 )
        {
          v22 = *(_QWORD *)(v53 + 88);
          goto LABEL_44;
        }
        if ( *(unsigned int *)(v53 + 100) < v18 )
        {
          sub_14B3F20(v53 + 88, v18);
          v21 = *(unsigned int *)(v53 + 96);
        }
        v22 = *(_QWORD *)(v53 + 88);
        v42 = v22 + 88 * v21;
        if ( v42 != v22 + v19 )
        {
          v50 = v18;
          v43 = v22 + v19;
          v47 = v18;
          v44 = v42;
          do
          {
            if ( v44 )
            {
              *(_QWORD *)v44 = v44 + 16;
              sub_15C7EA0((__int64 *)v44, "String", (__int64)"");
              *(_QWORD *)(v44 + 40) = 0;
              *(_QWORD *)(v44 + 32) = v44 + 48;
              *(_BYTE *)(v44 + 48) = 0;
              *(_QWORD *)(v44 + 64) = 0;
              *(_QWORD *)(v44 + 72) = 0;
              *(_DWORD *)(v44 + 80) = 0;
              *(_DWORD *)(v44 + 84) = 0;
            }
            v44 += 88;
          }
          while ( v43 != v44 );
LABEL_72:
          v20 = v50;
          v18 = v47;
          v22 = *(_QWORD *)(v53 + 88);
        }
      }
      *(_DWORD *)(v53 + 96) = v20;
      goto LABEL_44;
    }
  }
  return result;
}

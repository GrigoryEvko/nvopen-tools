// Function: sub_115CCA0
// Address: 0x115cca0
//
__int64 __fastcall sub_115CCA0(unsigned __int8 *a1, __int64 a2)
{
  bool v3; // zf
  int v4; // eax
  unsigned int v5; // eax
  __int64 v6; // rbx
  int v7; // r14d
  __int64 v8; // r13
  __int64 result; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  char v23; // r8
  int v24; // eax
  __int64 v25; // r13
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // r15
  char v36; // al
  unsigned int v37; // r15d
  __int64 v38; // rbx
  __int64 v39; // r14
  unsigned __int8 *v40; // r13
  __int64 *v41; // rdx
  __int64 v42; // rdx
  char v43; // al
  unsigned int v44; // r15d
  __int64 v45; // rbx
  __int64 v46; // r14
  unsigned __int8 *v47; // r13
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned int *v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rdx
  unsigned int v53; // esi
  unsigned int *v54; // rbx
  __int64 v55; // r14
  __int64 v56; // rdx
  unsigned int v57; // esi
  unsigned int *v58; // rbx
  __int64 v59; // r14
  __int64 v60; // rdx
  unsigned int v61; // esi
  bool v62; // [rsp+Fh] [rbp-E1h]
  bool v63; // [rsp+Fh] [rbp-E1h]
  __int64 v64; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v65; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v66; // [rsp+28h] [rbp-C8h]
  __int64 v67; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v68; // [rsp+50h] [rbp-A0h]
  _BYTE v69[32]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v70; // [rsp+80h] [rbp-70h]
  __int64 *v71; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v72; // [rsp+98h] [rbp-58h] BYREF
  unsigned __int64 v73; // [rsp+A0h] [rbp-50h] BYREF
  __int64 *v74; // [rsp+A8h] [rbp-48h]
  __int16 v75; // [rsp+B0h] [rbp-40h]

  v3 = *a1 == 46;
  v71 = &v64;
  v72 = 0;
  v73 = 0;
  v74 = &v65;
  if ( !v3 )
    goto LABEL_2;
  v10 = *((_QWORD *)a1 - 8);
  v11 = *(_QWORD *)(v10 + 16);
  if ( v11
    && !*(_QWORD *)(v11 + 8)
    && *(_BYTE *)v10 == 86
    && ((*(_BYTE *)(v10 + 7) & 0x40) == 0
      ? (v33 = (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)))
      : (v33 = *(__int64 **)(v10 - 8)),
        *v33
     && ((v64 = *v33, (*(_BYTE *)(v10 + 7) & 0x40) == 0)
       ? (v34 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF))
       : (v34 = *(_QWORD *)(v10 - 8)),
         (unsigned __int8)sub_993A50((_QWORD **)&v72, *(_QWORD *)(v34 + 32)))) )
  {
    if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
      v35 = *(_QWORD *)(v10 - 8);
    else
      v35 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
    v31 = *(_QWORD *)(v35 + 64);
    v36 = sub_995B10((_QWORD **)&v73, v31);
    v12 = *((_QWORD *)a1 - 4);
    if ( v36 && v12 )
    {
      *v74 = v12;
      goto LABEL_52;
    }
  }
  else
  {
    v12 = *((_QWORD *)a1 - 4);
  }
  v13 = *(_QWORD *)(v12 + 16);
  if ( v13 && !*(_QWORD *)(v13 + 8) && *(_BYTE *)v12 == 86 )
  {
    v28 = (*(_BYTE *)(v12 + 7) & 0x40) != 0
        ? *(__int64 **)(v12 - 8)
        : (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
    if ( *v28 )
    {
      *v71 = *v28;
      v29 = (*(_BYTE *)(v12 + 7) & 0x40) != 0 ? *(_QWORD *)(v12 - 8) : v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
      if ( (unsigned __int8)sub_993A50((_QWORD **)&v72, *(_QWORD *)(v29 + 32)) )
      {
        v30 = (*(_BYTE *)(v12 + 7) & 0x40) != 0 ? *(_QWORD *)(v12 - 8) : v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
        v31 = *(_QWORD *)(v30 + 64);
        if ( (unsigned __int8)sub_995B10((_QWORD **)&v73, v31) )
        {
          v32 = *((_QWORD *)a1 - 8);
          if ( v32 )
          {
            *v74 = v32;
LABEL_52:
            v37 = 1;
            v62 = sub_B44900((__int64)a1);
            if ( !v62 )
            {
              v62 = sub_B448F0((__int64)a1);
              v37 = v62;
            }
            v38 = v65;
            v70 = 257;
            v39 = sub_AD6530(*(_QWORD *)(v65 + 8), v31);
            v40 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a2 + 80) + 32LL))(
                                       *(_QWORD *)(a2 + 80),
                                       15,
                                       v39,
                                       v38,
                                       0,
                                       v37);
            if ( !v40 )
            {
              v75 = 257;
              v40 = (unsigned __int8 *)sub_B504D0(15, v39, v38, (__int64)&v71, 0, 0);
              (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
                *(_QWORD *)(a2 + 88),
                v40,
                v69,
                *(_QWORD *)(a2 + 56),
                *(_QWORD *)(a2 + 64));
              v54 = *(unsigned int **)a2;
              v55 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              if ( *(_QWORD *)a2 != v55 )
              {
                do
                {
                  v56 = *((_QWORD *)v54 + 1);
                  v57 = *v54;
                  v54 += 4;
                  sub_B99FD0((__int64)v40, v57, v56);
                }
                while ( (unsigned int *)v55 != v54 );
              }
              if ( v62 )
                sub_B44850(v40, 1);
            }
            v75 = 257;
            return sub_B36550((unsigned int **)a2, v64, v65, (__int64)v40, (__int64)&v71, 0);
          }
        }
      }
    }
  }
  v14 = *a1;
  v71 = &v64;
  v72 = 0;
  v73 = 0;
  v74 = &v65;
  if ( v14 != 46 )
    goto LABEL_2;
  v15 = *((_QWORD *)a1 - 8);
  v16 = *(_QWORD *)(v15 + 16);
  if ( !v16
    || *(_QWORD *)(v16 + 8)
    || *(_BYTE *)v15 != 86
    || ((*(_BYTE *)(v15 + 7) & 0x40) == 0
      ? (v41 = (__int64 *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF)))
      : (v41 = *(__int64 **)(v15 - 8)),
        !*v41
     || ((v64 = *v41, (*(_BYTE *)(v15 + 7) & 0x40) == 0)
       ? (v42 = v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF))
       : (v42 = *(_QWORD *)(v15 - 8)),
         !(unsigned __int8)sub_995B10((_QWORD **)&v72, *(_QWORD *)(v42 + 32)))) )
  {
    v17 = *((_QWORD *)a1 - 4);
    goto LABEL_12;
  }
  v21 = *(_QWORD *)(sub_986520(v15) + 64);
  v43 = sub_993A50((_QWORD **)&v73, v21);
  v17 = *((_QWORD *)a1 - 4);
  if ( !v43 || !v17 )
  {
LABEL_12:
    v18 = *(_QWORD *)(v17 + 16);
    if ( !v18
      || *(_QWORD *)(v18 + 8)
      || *(_BYTE *)v17 != 86
      || ((*(_BYTE *)(v17 + 7) & 0x40) == 0
        ? (v19 = (__int64 *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF)))
        : (v19 = *(__int64 **)(v17 - 8)),
          !*v19
       || ((*v71 = *v19, (*(_BYTE *)(v17 + 7) & 0x40) == 0)
         ? (v20 = v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF))
         : (v20 = *(_QWORD *)(v17 - 8)),
           !(unsigned __int8)sub_995B10((_QWORD **)&v72, *(_QWORD *)(v20 + 32))
        || (v21 = *(_QWORD *)(sub_986520(v17) + 64), !(unsigned __int8)sub_993A50((_QWORD **)&v73, v21))
        || (v22 = *((_QWORD *)a1 - 8)) == 0)) )
    {
LABEL_2:
      v71 = &v64;
      v74 = &v65;
      v72 = 0x3FF0000000000000LL;
      v73 = 0xBFF0000000000000LL;
      if ( (unsigned __int8)sub_115CA60((__int64)&v71, 18, a1) )
      {
        v70 = 257;
        v4 = sub_B45210((__int64)a1);
        BYTE4(v66) = 1;
        v68 = 257;
        LODWORD(v66) = v4;
        v5 = sub_B45210((__int64)a1);
        v6 = v65;
        v7 = v5;
        v8 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a2 + 80) + 48LL))(
               *(_QWORD *)(a2 + 80),
               12,
               v65,
               v5);
        if ( !v8 )
        {
          v75 = 257;
          v48 = sub_B50340(12, v6, (__int64)&v71, 0, 0);
          v49 = *(_QWORD *)(a2 + 96);
          v8 = v48;
          if ( v49 )
            sub_B99FD0(v48, 3u, v49);
          sub_B45150(v8, v7);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
            *(_QWORD *)(a2 + 88),
            v8,
            &v67,
            *(_QWORD *)(a2 + 56),
            *(_QWORD *)(a2 + 64));
          v50 = *(unsigned int **)a2;
          v51 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          if ( *(_QWORD *)a2 != v51 )
          {
            do
            {
              v52 = *((_QWORD *)v50 + 1);
              v53 = *v50;
              v50 += 4;
              sub_B99FD0(v8, v53, v52);
            }
            while ( (unsigned int *)v51 != v50 );
          }
        }
        return sub_B36280((unsigned int **)a2, v64, v65, v8, v66, (__int64)v69, 0);
      }
      else
      {
        v71 = &v64;
        v74 = &v65;
        v72 = 0xBFF0000000000000LL;
        v73 = 0x3FF0000000000000LL;
        v23 = sub_115CA60((__int64)&v71, 18, a1);
        result = 0;
        if ( v23 )
        {
          v75 = 257;
          v24 = sub_B45210((__int64)a1);
          v25 = v65;
          v70 = 257;
          LODWORD(v67) = v24;
          BYTE4(v67) = 1;
          v26 = sub_B45210((__int64)a1);
          v27 = sub_11553A0((__int64 *)a2, v25, v26, 1, (__int64)v69, 0);
          return sub_B36280((unsigned int **)a2, v64, v27, v25, v67, (__int64)&v71, 0);
        }
      }
      return result;
    }
    *v74 = v22;
    goto LABEL_67;
  }
  *v74 = v17;
LABEL_67:
  v44 = 1;
  v63 = sub_B44900((__int64)a1);
  if ( !v63 )
  {
    v63 = sub_B448F0((__int64)a1);
    v44 = v63;
  }
  v45 = v65;
  v70 = 257;
  v46 = sub_AD6530(*(_QWORD *)(v65 + 8), v21);
  v47 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a2 + 80) + 32LL))(
                             *(_QWORD *)(a2 + 80),
                             15,
                             v46,
                             v45,
                             0,
                             v44);
  if ( !v47 )
  {
    v75 = 257;
    v47 = (unsigned __int8 *)sub_B504D0(15, v46, v45, (__int64)&v71, 0, 0);
    (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v47,
      v69,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v58 = *(unsigned int **)a2;
    v59 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v59 )
    {
      do
      {
        v60 = *((_QWORD *)v58 + 1);
        v61 = *v58;
        v58 += 4;
        sub_B99FD0((__int64)v47, v61, v60);
      }
      while ( (unsigned int *)v59 != v58 );
    }
    if ( v63 )
      sub_B44850(v47, 1);
  }
  v75 = 257;
  return sub_B36550((unsigned int **)a2, v64, (__int64)v47, v65, (__int64)&v71, 0);
}

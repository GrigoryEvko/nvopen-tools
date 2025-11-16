// Function: sub_1EBE5E0
// Address: 0x1ebe5e0
//
__int64 __fastcall sub_1EBE5E0(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // r13d
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdi
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 (*v16)(); // rcx
  __int64 v17; // rdx
  unsigned int v18; // edx
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rdi
  signed __int64 v24; // r9
  unsigned __int64 v25; // r8
  int v26; // r9d
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rax
  _QWORD *v29; // rdx
  unsigned int v30; // eax
  unsigned int v31; // r13d
  __int64 v32; // r15
  __int64 v33; // rbx
  __int64 v34; // rdx
  _DWORD *v35; // rax
  __int64 v36; // r14
  int v37; // r9d
  unsigned __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // r13
  __int64 v41; // r10
  unsigned int v42; // ebx
  __int64 v43; // rcx
  __int64 v45; // rdx
  _QWORD *v46; // rcx
  _QWORD *i; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  unsigned __int64 v50; // rsi
  int v51; // r14d
  _QWORD *v52; // rcx
  _QWORD *v53; // rax
  _QWORD *v54; // rsi
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // [rsp+8h] [rbp-178h]
  int v59; // [rsp+14h] [rbp-16Ch]
  unsigned __int64 v60; // [rsp+18h] [rbp-168h]
  __int64 v61; // [rsp+28h] [rbp-158h]
  __int64 v62; // [rsp+38h] [rbp-148h]
  unsigned __int64 v63[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v64[32]; // [rsp+50h] [rbp-130h] BYREF
  _QWORD v65[2]; // [rsp+70h] [rbp-110h] BYREF
  __int64 v66; // [rsp+80h] [rbp-100h]
  __int64 v67; // [rsp+88h] [rbp-F8h]
  __int64 v68; // [rsp+90h] [rbp-F0h]
  __int64 v69; // [rsp+98h] [rbp-E8h]
  __int64 v70; // [rsp+A0h] [rbp-E0h]
  __int64 v71; // [rsp+A8h] [rbp-D8h]
  unsigned int v72; // [rsp+B0h] [rbp-D0h]
  char v73; // [rsp+B4h] [rbp-CCh]
  __int64 v74; // [rsp+B8h] [rbp-C8h]
  __int64 v75; // [rsp+C0h] [rbp-C0h]
  _BYTE *v76; // [rsp+C8h] [rbp-B8h]
  _BYTE *v77; // [rsp+D0h] [rbp-B0h]
  __int64 v78; // [rsp+D8h] [rbp-A8h]
  int v79; // [rsp+E0h] [rbp-A0h]
  _BYTE v80[32]; // [rsp+E8h] [rbp-98h] BYREF
  __int64 v81; // [rsp+108h] [rbp-78h]
  _BYTE *v82; // [rsp+110h] [rbp-70h]
  _BYTE *v83; // [rsp+118h] [rbp-68h]
  __int64 v84; // [rsp+120h] [rbp-60h]
  int v85; // [rsp+128h] [rbp-58h]
  _BYTE v86[80]; // [rsp+130h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a2 + 112);
  v7 = *(_QWORD *)(a1 + 280)
     + 24LL
     * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16LL
                                                                                               * (v6 & 0x7FFFFFFF))
                                       & 0xFFFFFFFFFFFFFFF8LL)
                           + 24LL);
  if ( *(_DWORD *)(a1 + 288) != *(_DWORD *)v7 )
    sub_1ED7890(a1 + 280);
  v8 = *(_QWORD *)(a1 + 680);
  v9 = a1 + 376;
  v10 = a1 + 672;
  v11 = *(_QWORD *)(a1 + 256);
  v12 = *(_QWORD *)(a1 + 264);
  v13 = *(unsigned __int8 *)(v7 + 8);
  v66 = a3;
  v65[0] = &unk_4A00C10;
  v14 = *(_QWORD *)(v8 + 40);
  v68 = v12;
  v69 = v11;
  v67 = v14;
  v15 = *(_QWORD *)(v8 + 16);
  v65[1] = a2;
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 40LL);
  v17 = 0;
  if ( v16 != sub_1D00B00 )
  {
    v57 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v16)(v15, v9, 0);
    v10 = a1 + 672;
    v9 = a1 + 376;
    v17 = v57;
    v14 = v67;
  }
  v70 = v17;
  v18 = *(_DWORD *)(a3 + 8);
  v74 = v9;
  v72 = v18;
  v76 = v80;
  v77 = v80;
  v82 = v86;
  v83 = v86;
  v71 = v10;
  v73 = 0;
  v75 = 0;
  v78 = 4;
  v79 = 0;
  v81 = 0;
  v84 = 4;
  v85 = 0;
  *(_QWORD *)(v14 + 8) = v65;
  sub_1F15B50(*(_QWORD *)(a1 + 992), v65, (unsigned int)dword_4FC99C0);
  v19 = *(_QWORD *)(a1 + 984);
  v20 = *(_DWORD *)(v19 + 288);
  v21 = *(_QWORD *)(v19 + 280);
  if ( v20 )
  {
    v22 = v21 + 40LL * (unsigned int)(v20 - 1);
    while ( 1 )
    {
      if ( (unsigned __int8)sub_1F15AD0(v19, v21, v13) )
      {
        sub_1F203D0(*(_QWORD *)(a1 + 992), v21);
        if ( v22 == v21 )
          break;
      }
      else if ( v22 == v21 )
      {
        break;
      }
      v19 = *(_QWORD *)(a1 + 984);
      v21 += 40;
    }
  }
  if ( *(_DWORD *)(v66 + 8) == v72 )
    goto LABEL_29;
  v23 = *(_QWORD *)(a1 + 992);
  v63[0] = (unsigned __int64)v64;
  v63[1] = 0x800000000LL;
  sub_1F1E080(v23, v63);
  sub_1DADA60(
    *(_QWORD *)(a1 + 856),
    v6,
    *(_QWORD *)v66 + 4LL * v72,
    *(unsigned int *)(v66 + 8) - (unsigned __int64)v72,
    *(_QWORD *)(a1 + 264),
    v24);
  v27 = *(unsigned int *)(*(_QWORD *)(a1 + 248) + 32LL);
  v61 = a1 + 920;
  v28 = *(unsigned int *)(a1 + 928);
  if ( v27 >= v28 )
  {
    if ( v27 <= v28 )
      goto LABEL_14;
    if ( v27 > *(unsigned int *)(a1 + 932) )
    {
      sub_16CD150(v61, (const void *)(a1 + 936), v27, 8, v25, v26);
      v28 = *(unsigned int *)(a1 + 928);
    }
    v45 = *(_QWORD *)(a1 + 920);
    v46 = (_QWORD *)(v45 + 8 * v27);
    for ( i = (_QWORD *)(v45 + 8 * v28); v46 != i; ++i )
    {
      if ( i )
        *i = *(_QWORD *)(a1 + 936);
    }
  }
  *(_DWORD *)(a1 + 928) = v27;
LABEL_14:
  v29 = (_QWORD *)v66;
  v30 = v72;
  v31 = *(_DWORD *)(v66 + 8) - v72;
  if ( v31 )
  {
    v32 = 0;
    v62 = v31;
    while ( 1 )
    {
      v36 = *(_QWORD *)(a1 + 264);
      v37 = *(_DWORD *)(*v29 + 4LL * (v30 + (unsigned int)v32));
      v38 = *(unsigned int *)(v36 + 408);
      v39 = v37 & 0x7FFFFFFF;
      v40 = v37 & 0x7FFFFFFF;
      v41 = 8 * v40;
      if ( (v37 & 0x7FFFFFFFu) >= (unsigned int)v38 )
        break;
      v33 = *(_QWORD *)(*(_QWORD *)(v36 + 400) + 8LL * v39);
      if ( !v33 )
        break;
LABEL_17:
      v34 = *(_QWORD *)(a1 + 920);
      v35 = (_DWORD *)(v34 + 8LL * (*(_DWORD *)(v33 + 112) & 0x7FFFFFFF));
      if ( !*v35 && !*(_DWORD *)(v63[0] + 4 * v32) )
      {
        v49 = *(_QWORD *)(a1 + 248);
        v50 = *(unsigned int *)(a1 + 928);
        v25 = *(unsigned int *)(v49 + 32);
        v51 = *(_DWORD *)(v49 + 32);
        if ( v25 < v50 )
          goto LABEL_53;
        if ( v25 > v50 )
        {
          if ( v25 > *(unsigned int *)(a1 + 932) )
          {
            v60 = *(unsigned int *)(v49 + 32);
            sub_16CD150(v61, (const void *)(a1 + 936), v60, 8, v25, v37);
            v34 = *(_QWORD *)(a1 + 920);
            v50 = *(unsigned int *)(a1 + 928);
            v25 = v60;
          }
          v52 = (_QWORD *)(v34 + 8 * v25);
          v53 = (_QWORD *)(v34 + 8 * v50);
          if ( v52 != v53 )
          {
            do
            {
              if ( v53 )
                *v53 = *(_QWORD *)(a1 + 936);
              ++v53;
            }
            while ( v52 != v53 );
            v34 = *(_QWORD *)(a1 + 920);
          }
LABEL_53:
          *(_DWORD *)(a1 + 928) = v51;
          v35 = (_DWORD *)(v34 + 8LL * (*(_DWORD *)(v33 + 112) & 0x7FFFFFFF));
        }
        *v35 = 4;
      }
      if ( ++v32 == v62 )
        goto LABEL_25;
      v30 = v72;
      v29 = (_QWORD *)v66;
    }
    v42 = v39 + 1;
    if ( (unsigned int)v38 < v39 + 1 )
    {
      v48 = v42;
      if ( v42 >= v38 )
      {
        if ( v42 > v38 )
        {
          if ( v42 > (unsigned __int64)*(unsigned int *)(v36 + 412) )
          {
            v58 = 8LL * (v37 & 0x7FFFFFFF);
            v59 = v37;
            sub_16CD150(v36 + 400, (const void *)(v36 + 416), v42, 8, v25, v37);
            v38 = *(unsigned int *)(v36 + 408);
            v41 = v58;
            v37 = v59;
            v48 = v42;
          }
          v43 = *(_QWORD *)(v36 + 400);
          v54 = (_QWORD *)(v43 + 8 * v48);
          v55 = (_QWORD *)(v43 + 8 * v38);
          v56 = *(_QWORD *)(v36 + 416);
          if ( v54 != v55 )
          {
            do
              *v55++ = v56;
            while ( v54 != v55 );
            v43 = *(_QWORD *)(v36 + 400);
          }
          *(_DWORD *)(v36 + 408) = v42;
          goto LABEL_24;
        }
      }
      else
      {
        *(_DWORD *)(v36 + 408) = v42;
      }
    }
    v43 = *(_QWORD *)(v36 + 400);
LABEL_24:
    *(_QWORD *)(v43 + v41) = sub_1DBA290(v37);
    v33 = *(_QWORD *)(*(_QWORD *)(v36 + 400) + 8 * v40);
    sub_1DBB110((_QWORD *)v36, v33);
    goto LABEL_17;
  }
LABEL_25:
  if ( byte_4FCF965[0] )
    sub_1E926D0(*(_QWORD *)(a1 + 680), a1, (__int64)"After splitting live range around basic blocks", 1);
  if ( (_BYTE *)v63[0] != v64 )
    _libc_free(v63[0]);
LABEL_29:
  *(_QWORD *)(v67 + 8) = 0;
  if ( v83 != v82 )
    _libc_free((unsigned __int64)v83);
  if ( v77 != v76 )
    _libc_free((unsigned __int64)v77);
  return 0;
}

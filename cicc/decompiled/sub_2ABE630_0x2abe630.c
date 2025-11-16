// Function: sub_2ABE630
// Address: 0x2abe630
//
unsigned __int8 *__fastcall sub_2ABE630(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 *a6)
{
  __int64 v9; // r11
  __int64 v10; // r15
  unsigned __int8 *v11; // r15
  __int64 *v12; // rdx
  bool v13; // cc
  unsigned int v14; // ebx
  bool v15; // al
  bool v16; // zf
  __int64 v17; // r14
  __int64 v18; // rdi
  unsigned int v19; // r15d
  __int64 v20; // rax
  unsigned __int8 *v22; // r14
  unsigned int v23; // ebx
  bool v24; // al
  unsigned int v25; // eax
  __int64 v26; // rax
  unsigned __int8 *v27; // rax
  __int64 v28; // rdx
  int v29; // r8d
  unsigned int *v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  unsigned int v34; // ebx
  __int64 v35; // rdi
  unsigned int *v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // rdx
  int v41; // r8d
  unsigned int *v42; // rax
  int v43; // r14d
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r15
  __int64 v47; // r12
  __int64 v48; // rdx
  int v49; // r14d
  bool v50; // bl
  unsigned int v51; // r15d
  __int64 v52; // rax
  unsigned int v53; // ebx
  unsigned int v55; // [rsp+8h] [rbp-C8h]
  __int64 v56; // [rsp+8h] [rbp-C8h]
  __int64 v57; // [rsp+8h] [rbp-C8h]
  __int64 v59; // [rsp+18h] [rbp-B8h]
  __int64 v60; // [rsp+18h] [rbp-B8h]
  int v61; // [rsp+18h] [rbp-B8h]
  unsigned int *v62; // [rsp+18h] [rbp-B8h]
  __int64 v63; // [rsp+18h] [rbp-B8h]
  __int64 v64; // [rsp+18h] [rbp-B8h]
  int v65; // [rsp+18h] [rbp-B8h]
  unsigned int *v66; // [rsp+18h] [rbp-B8h]
  int v67; // [rsp+18h] [rbp-B8h]
  unsigned int **v68; // [rsp+28h] [rbp-A8h] BYREF
  unsigned int v69; // [rsp+30h] [rbp-A0h]
  int v70; // [rsp+34h] [rbp-9Ch]
  __int64 v71; // [rsp+38h] [rbp-98h]
  _QWORD v72[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v73; // [rsp+60h] [rbp-70h]
  __int64 *v74[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v75; // [rsp+90h] [rbp-40h]

  v9 = *(_QWORD *)(a4 + 8);
  v10 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v9 + 8) != 12 )
  {
    v73 = 257;
    if ( v9 != v10 )
    {
      v59 = v9;
      v11 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80)
                                                                                            + 120LL))(
                                 *(_QWORD *)(a1 + 80),
                                 44,
                                 a2,
                                 v9);
      if ( !v11 )
      {
        v75 = 257;
        v11 = (unsigned __int8 *)sub_B51D30(44, a2, v59, (__int64)v74, 0, 0);
        if ( (unsigned __int8)sub_920620((__int64)v11) )
        {
          v28 = *(_QWORD *)(a1 + 96);
          v29 = *(_DWORD *)(a1 + 104);
          if ( v28 )
          {
            v61 = *(_DWORD *)(a1 + 104);
            sub_B99FD0((__int64)v11, 3u, v28);
            v29 = v61;
          }
          sub_B45150((__int64)v11, v29);
        }
        (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v11,
          v72,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v30 = *(unsigned int **)a1;
        v56 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v56 )
        {
          do
          {
            v62 = v30;
            sub_B99FD0((__int64)v11, *v30, *((_QWORD *)v30 + 1));
            v30 = v62 + 4;
          }
          while ( (unsigned int *)v56 != v62 + 4 );
        }
      }
      goto LABEL_4;
    }
LABEL_32:
    v68 = (unsigned int **)a1;
    v11 = (unsigned __int8 *)a2;
    v13 = a5 <= 2;
    if ( a5 != 2 )
      goto LABEL_7;
LABEL_33:
    v75 = 257;
    v27 = sub_2AAE1E0(&v68, (__int64)v11, a4);
    return (unsigned __int8 *)sub_F7CA10((__int64 *)a1, a3, (__int64)v27, (__int64)v74, 0);
  }
  v60 = *(_QWORD *)(a4 + 8);
  v73 = 257;
  v55 = sub_BCB060(v10);
  v25 = sub_BCB060(v60);
  if ( v55 < v25 )
  {
    if ( v60 == v10 )
    {
      v11 = (unsigned __int8 *)a2;
    }
    else
    {
      v11 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80)
                                                                                            + 120LL))(
                                 *(_QWORD *)(a1 + 80),
                                 40,
                                 a2,
                                 v60);
      if ( !v11 )
      {
        v75 = 257;
        v11 = (unsigned __int8 *)sub_B51D30(40, a2, v60, (__int64)v74, 0, 0);
        if ( (unsigned __int8)sub_920620((__int64)v11) )
        {
          v40 = *(_QWORD *)(a1 + 96);
          v41 = *(_DWORD *)(a1 + 104);
          if ( v40 )
          {
            v65 = *(_DWORD *)(a1 + 104);
            sub_B99FD0((__int64)v11, 3u, v40);
            v41 = v65;
          }
          sub_B45150((__int64)v11, v41);
        }
        (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v11,
          v72,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v42 = *(unsigned int **)a1;
        v57 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v57 )
        {
          do
          {
            v66 = v42;
            sub_B99FD0((__int64)v11, *v42, *((_QWORD *)v42 + 1));
            v42 = v66 + 4;
          }
          while ( (unsigned int *)v57 != v66 + 4 );
        }
      }
    }
  }
  else
  {
    if ( v60 == v10 || v55 == v25 )
      goto LABEL_32;
    v11 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80)
                                                                                          + 120LL))(
                               *(_QWORD *)(a1 + 80),
                               38,
                               a2,
                               v60);
    if ( !v11 )
    {
      v75 = 257;
      v26 = sub_B51D30(38, a2, v60, (__int64)v74, 0, 0);
      v11 = (unsigned __int8 *)sub_289B9A0((__int64 *)a1, v26, (__int64)v72);
    }
  }
LABEL_4:
  if ( v11 != (unsigned __int8 *)a2 )
  {
    v74[0] = (__int64 *)sub_BD5D20((__int64)v11);
    v75 = 773;
    v74[1] = v12;
    v74[2] = (__int64 *)".cast";
    sub_BD6B50(v11, (const char **)v74);
  }
  v68 = (unsigned int **)a1;
  v13 = a5 <= 2;
  if ( a5 == 2 )
    goto LABEL_33;
LABEL_7:
  if ( !v13 )
  {
    if ( a5 != 3 )
      BUG();
    v70 = 0;
    v16 = *(_BYTE *)(a1 + 108) == 0;
    v73 = 257;
    v71 = v69;
    if ( v16 )
    {
      v17 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, unsigned __int8 *, _QWORD))(**(_QWORD **)(a1 + 80)
                                                                                           + 40LL))(
              *(_QWORD *)(a1 + 80),
              18,
              a4,
              v11,
              *(unsigned int *)(a1 + 104));
      if ( !v17 )
      {
        v75 = 257;
        v43 = *(_DWORD *)(a1 + 104);
        v44 = sub_B504D0(18, a4, (__int64)v11, (__int64)v74, 0, 0);
        v45 = *(_QWORD *)(a1 + 96);
        v46 = v44;
        if ( v45 )
          sub_B99FD0(v44, 3u, v45);
        sub_B45150(v46, v43);
        v17 = sub_289B9A0((__int64 *)a1, v46, (__int64)v72);
      }
    }
    else
    {
      v17 = sub_B35400(a1, 0x6Cu, a4, (__int64)v11, v69, (__int64)v72, 0, 0, 0);
    }
    v18 = *(_QWORD *)(a1 + 80);
    v73 = 259;
    v72[0] = "induction";
    v19 = *a6 - 29;
    v20 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v18 + 16LL))(v18, v19, a3, v17);
    if ( !v20 )
    {
      v75 = 257;
      v47 = sub_B504D0(v19, a3, v17, (__int64)v74, 0, 0);
      if ( (unsigned __int8)sub_920620(v47) )
      {
        v48 = *(_QWORD *)(a1 + 96);
        v49 = *(_DWORD *)(a1 + 104);
        if ( v48 )
          sub_B99FD0(v47, 3u, v48);
        sub_B45150(v47, v49);
      }
      return (unsigned __int8 *)sub_289B9A0((__int64 *)a1, v47, (__int64)v72);
    }
    return (unsigned __int8 *)v20;
  }
  if ( !a5 )
    return 0;
  if ( *(_BYTE *)a4 != 17
    || (v14 = *(_DWORD *)(a4 + 32)) != 0
    && (v14 <= 0x40
      ? (v15 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) == *(_QWORD *)(a4 + 24))
      : (v15 = v14 == (unsigned int)sub_C445E0(a4 + 24)),
        !v15) )
  {
    v22 = sub_2AAE1E0(&v68, (__int64)v11, a4);
    if ( *(_BYTE *)a3 == 17 )
    {
      v23 = *(_DWORD *)(a3 + 32);
      if ( v23 <= 0x40 )
        v24 = *(_QWORD *)(a3 + 24) == 0;
      else
        v24 = v23 == (unsigned int)sub_C444A0(a3 + 24);
      if ( v24 )
        return v22;
      goto LABEL_47;
    }
    v31 = *(_QWORD *)(a3 + 8);
    v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
    if ( (unsigned int)v32 <= 1 && *(_BYTE *)a3 <= 0x15u )
    {
      v33 = sub_AD7630(a3, 0, v32);
      if ( !v33 || *v33 != 17 )
      {
        if ( *(_BYTE *)(v31 + 8) == 17 )
        {
          v67 = *(_DWORD *)(v31 + 32);
          if ( v67 )
          {
            v50 = 0;
            v51 = 0;
            while ( 1 )
            {
              v52 = sub_AD69F0((unsigned __int8 *)a3, v51);
              if ( !v52 )
                break;
              if ( *(_BYTE *)v52 != 13 )
              {
                if ( *(_BYTE *)v52 != 17 )
                  break;
                v53 = *(_DWORD *)(v52 + 32);
                v50 = v53 <= 0x40 ? *(_QWORD *)(v52 + 24) == 0 : v53 == (unsigned int)sub_C444A0(v52 + 24);
                if ( !v50 )
                  break;
              }
              if ( v67 == ++v51 )
              {
                if ( v50 )
                  return v22;
                goto LABEL_47;
              }
            }
          }
        }
        goto LABEL_47;
      }
      v34 = *((_DWORD *)v33 + 8);
      if ( v34 <= 0x40 )
      {
        if ( *((_QWORD *)v33 + 3) )
          goto LABEL_47;
      }
      else if ( v34 != (unsigned int)sub_C444A0((__int64)(v33 + 24)) )
      {
        goto LABEL_47;
      }
      return v22;
    }
LABEL_47:
    v74[0] = 0;
    if ( (unsigned __int8)sub_10081F0(v74, (__int64)v22) )
      return (unsigned __int8 *)a3;
    v35 = *(_QWORD *)(a1 + 80);
    v73 = 257;
    v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD, _QWORD))(*(_QWORD *)v35 + 32LL))(
            v35,
            13,
            a3,
            v22,
            0,
            0);
    if ( !v20 )
    {
      v75 = 257;
      v63 = sub_B504D0(13, a3, (__int64)v22, (__int64)v74, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v63,
        v72,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v36 = *(unsigned int **)a1;
      v20 = v63;
      v37 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v37 )
      {
        do
        {
          v38 = *((_QWORD *)v36 + 1);
          v39 = *v36;
          v36 += 4;
          v64 = v20;
          sub_B99FD0(v20, v39, v38);
          v20 = v64;
        }
        while ( (unsigned int *)v37 != v36 );
      }
    }
    return (unsigned __int8 *)v20;
  }
  v75 = 257;
  return (unsigned __int8 *)sub_929DE0((unsigned int **)a1, (_BYTE *)a3, v11, (__int64)v74, 0, 0);
}

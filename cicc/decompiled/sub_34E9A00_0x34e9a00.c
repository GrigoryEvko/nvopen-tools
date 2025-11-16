// Function: sub_34E9A00
// Address: 0x34e9a00
//
__int64 __fastcall sub_34E9A00(
        __int64 a1,
        __int64 *a2,
        __int64 *a3,
        _QWORD *a4,
        _QWORD *a5,
        _DWORD *a6,
        _DWORD *a7,
        __int64 a8,
        __int64 a9,
        char a10)
{
  _QWORD *v11; // rcx
  _QWORD *v12; // rax
  __int64 i; // rsi
  _QWORD *v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdi
  __int64 (*v22)(); // rax
  int v23; // eax
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  _QWORD *v27; // rdx
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // r8
  __int64 v38; // rax
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rax
  int v41; // eax
  bool v42; // al
  int v43; // eax
  bool v44; // al
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  int v53; // eax
  __int64 v54; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // rax
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  int v62; // eax
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  __int64 v68; // rax
  unsigned __int64 v69; // rax
  int v70; // eax
  __int64 v71; // rax
  int v72; // eax
  __int64 v73; // rax
  _QWORD *v74; // rax
  _QWORD *v75; // rdx
  __int64 v76; // rax
  unsigned __int64 v77; // rax
  unsigned __int64 v79; // [rsp+18h] [rbp-58h]
  unsigned __int64 v80; // [rsp+18h] [rbp-58h]
  unsigned __int64 v81; // [rsp+18h] [rbp-58h]
  unsigned __int64 v82; // [rsp+18h] [rbp-58h]
  unsigned __int64 v83; // [rsp+18h] [rbp-58h]
  unsigned __int64 v84; // [rsp+18h] [rbp-58h]
  unsigned __int64 v85; // [rsp+18h] [rbp-58h]
  unsigned __int64 v86[10]; // [rsp+20h] [rbp-50h] BYREF

  v11 = (_QWORD *)*a4;
  v12 = (_QWORD *)*a2;
  if ( (_QWORD *)*a2 != v11 )
  {
    while ( 1 )
    {
      if ( *a5 == *a3 )
        return 1;
      for ( ; v12 != v11; v12 = (_QWORD *)v12[1] )
      {
        if ( (unsigned __int16)(*((_WORD *)v12 + 34) - 14) > 4u )
          break;
        if ( (*(_BYTE *)v12 & 4) == 0 )
        {
          while ( (*((_BYTE *)v12 + 44) & 8) != 0 )
            v12 = (_QWORD *)v12[1];
        }
      }
      *a2 = (__int64)v12;
      for ( i = *a3; *a5 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( (unsigned __int16)(*(_WORD *)(i + 68) - 14) > 4u )
          break;
        if ( (*(_BYTE *)i & 4) == 0 )
        {
          while ( (*(_BYTE *)(i + 44) & 8) != 0 )
            i = *(_QWORD *)(i + 8);
        }
      }
      *a3 = i;
      v18 = (_QWORD *)*a2;
      v19 = (_QWORD *)*a4;
      if ( *a4 == *a2 )
        return 1;
      if ( i == *a5 )
        goto LABEL_44;
      if ( !sub_2E88AF0((__int64)v18, i, 0) )
      {
        v19 = (_QWORD *)*a4;
        v18 = (_QWORD *)*a2;
        if ( *a2 == *a4 )
          return 1;
LABEL_44:
        v27 = (_QWORD *)*a3;
        if ( *a3 == *a5 )
          return 1;
        v28 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v28 )
          BUG();
        v29 = *(_QWORD *)v28;
        if ( (*(_QWORD *)v28 & 4) == 0 && (*(_BYTE *)(v28 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
            v28 = v30;
            if ( (*(_BYTE *)(v30 + 44) & 4) == 0 )
              break;
            v29 = *(_QWORD *)v30;
          }
        }
        v31 = *(_QWORD *)*a5 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v31 )
          BUG();
        v32 = *(_QWORD *)v31;
        if ( (*(_QWORD *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v33 = v32 & 0xFFFFFFFFFFFFFFF8LL;
            v31 = v33;
            if ( (*(_BYTE *)(v33 + 44) & 4) == 0 )
              break;
            v32 = *(_QWORD *)v33;
          }
        }
        v34 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v34 )
          BUG();
        v35 = *(_QWORD *)v34;
        if ( (*(_QWORD *)v34 & 4) == 0 && (*(_BYTE *)(v34 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v36 = v35 & 0xFFFFFFFFFFFFFFF8LL;
            v34 = v36;
            if ( (*(_BYTE *)(v36 + 44) & 4) == 0 )
              break;
            v35 = *(_QWORD *)v36;
          }
        }
        v37 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v37 )
          BUG();
        v38 = *(_QWORD *)v37;
        v39 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v37 & 4) == 0 && (*(_BYTE *)(v37 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v40 = v38 & 0xFFFFFFFFFFFFFFF8LL;
            v39 = v40;
            if ( (*(_BYTE *)(v40 + 44) & 4) == 0 )
              break;
            v38 = *(_QWORD *)v40;
          }
        }
        if ( (*(_DWORD *)(a8 + 120) || *(_DWORD *)(a9 + 120)) && a10 )
        {
          if ( v34 != v28 )
          {
            do
            {
              v41 = *(_DWORD *)(v28 + 44);
              if ( (v41 & 4) != 0 || (v41 & 8) == 0 )
              {
                if ( (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) & 0x400LL) == 0 )
                  break;
              }
              else
              {
                v79 = v34;
                v42 = sub_2E88A90(v28, 1024, 1);
                v34 = v79;
                if ( !v42 )
                  break;
              }
              v70 = *(_DWORD *)(v28 + 44);
              if ( (v70 & 4) == 0 && (v70 & 8) != 0 )
              {
                v82 = v34;
                LOBYTE(v71) = sub_2E88A90(v28, 256, 1);
                v34 = v82;
              }
              else
              {
                v71 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) >> 8) & 1LL;
              }
              if ( !(_BYTE)v71 )
                break;
              v72 = *(_DWORD *)(v28 + 44);
              if ( (v72 & 4) != 0 || (v72 & 8) == 0 )
              {
                v73 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) >> 11) & 1LL;
              }
              else
              {
                v84 = v34;
                LOBYTE(v73) = sub_2E88A90(v28, 2048, 1);
                v34 = v84;
              }
              if ( (_BYTE)v73 )
                break;
              v74 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
              v75 = v74;
              if ( !v74 )
                BUG();
              v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
              v76 = *v74;
              if ( (v76 & 4) == 0 && (*((_BYTE *)v75 + 44) & 4) != 0 )
              {
                while ( 1 )
                {
                  v77 = v76 & 0xFFFFFFFFFFFFFFF8LL;
                  v28 = v77;
                  if ( (*(_BYTE *)(v77 + 44) & 4) == 0 )
                    break;
                  v76 = *(_QWORD *)v77;
                }
              }
            }
            while ( v34 != v28 );
            while ( v31 != v39 )
            {
LABEL_76:
              v43 = *(_DWORD *)(v31 + 44);
              if ( (v43 & 4) != 0 || (v43 & 8) == 0 )
              {
                if ( (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 24LL) & 0x400LL) == 0 )
                  goto LABEL_79;
              }
              else
              {
                v80 = v34;
                v44 = sub_2E88A90(v31, 1024, 1);
                v34 = v80;
                if ( !v44 )
                  goto LABEL_79;
              }
              v62 = *(_DWORD *)(v31 + 44);
              if ( (v62 & 4) == 0 && (v62 & 8) != 0 )
              {
                v83 = v34;
                LOBYTE(v63) = sub_2E88A90(v31, 256, 1);
                v34 = v83;
              }
              else
              {
                v63 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 24LL) >> 8) & 1LL;
              }
              if ( !(_BYTE)v63 )
                break;
              v64 = *(_DWORD *)(v31 + 44);
              if ( (v64 & 4) != 0 || (v64 & 8) == 0 )
              {
                v65 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 24LL) >> 11) & 1LL;
              }
              else
              {
                v85 = v34;
                LOBYTE(v65) = sub_2E88A90(v31, 2048, 1);
                v34 = v85;
              }
              if ( (_BYTE)v65 )
                break;
              v66 = (_QWORD *)(*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL);
              v67 = v66;
              if ( !v66 )
                BUG();
              v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
              v68 = *v66;
              if ( (v68 & 4) == 0 && (*((_BYTE *)v67 + 44) & 4) != 0 )
              {
                while ( 1 )
                {
                  v69 = v68 & 0xFFFFFFFFFFFFFFF8LL;
                  v31 = v69;
                  if ( (*(_BYTE *)(v69 + 44) & 4) == 0 )
                    break;
                  v68 = *(_QWORD *)v69;
                }
              }
            }
            goto LABEL_79;
          }
          if ( v39 != v31 )
            goto LABEL_76;
LABEL_101:
          if ( (*(_BYTE *)v28 & 4) != 0 )
          {
LABEL_102:
            *a4 = *(_QWORD *)(v28 + 8);
            if ( !v31 )
              BUG();
            if ( (*(_BYTE *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 44) & 8) != 0 )
            {
              do
                v31 = *(_QWORD *)(v31 + 8);
              while ( (*(_BYTE *)(v31 + 44) & 8) != 0 );
            }
            *a5 = *(_QWORD *)(v31 + 8);
            return 1;
          }
        }
        else
        {
LABEL_79:
          if ( v28 == v34 )
            goto LABEL_101;
          do
          {
            if ( v31 == v39 )
              goto LABEL_100;
            while ( v34 != v28 && (unsigned __int16)(*(_WORD *)(v28 + 68) - 14) <= 4u )
            {
              v45 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
              v46 = v45;
              if ( !v45 )
                BUG();
              v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
              v47 = *v45;
              if ( (v47 & 4) == 0 && (*((_BYTE *)v46 + 44) & 4) != 0 )
              {
                while ( 1 )
                {
                  v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
                  v28 = v48;
                  if ( (*(_BYTE *)(v48 + 44) & 4) == 0 )
                    break;
                  v47 = *(_QWORD *)v48;
                }
              }
            }
            while ( (unsigned __int16)(*(_WORD *)(v31 + 68) - 14) <= 4u )
            {
              v49 = (_QWORD *)(*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL);
              v50 = v49;
              if ( !v49 )
                BUG();
              v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
              v51 = *v49;
              if ( (v51 & 4) == 0 && (*((_BYTE *)v50 + 44) & 4) != 0 )
              {
                while ( 1 )
                {
                  v52 = v51 & 0xFFFFFFFFFFFFFFF8LL;
                  v31 = v52;
                  if ( (*(_BYTE *)(v52 + 44) & 4) == 0 )
                    break;
                  v51 = *(_QWORD *)v52;
                }
                if ( v52 == v39 )
                  goto LABEL_100;
              }
              else if ( v31 == v39 )
              {
                goto LABEL_100;
              }
            }
            v81 = v34;
            if ( v28 == v34 || v31 == v39 || !sub_2E88AF0(v28, v31, 0) )
            {
LABEL_100:
              if ( !v28 )
                BUG();
              goto LABEL_101;
            }
            v53 = *(_DWORD *)(v28 + 44);
            v34 = v81;
            if ( (v53 & 4) != 0 || (v53 & 8) == 0 )
            {
              v54 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) >> 10) & 1LL;
            }
            else
            {
              LOBYTE(v54) = sub_2E88A90(v28, 1024, 1);
              v34 = v81;
            }
            if ( !(_BYTE)v54 )
              ++*a7;
            v55 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
            v56 = v55;
            if ( !v55 )
              BUG();
            v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
            v57 = *v55;
            if ( (v57 & 4) == 0 && (*((_BYTE *)v56 + 44) & 4) != 0 )
            {
              while ( 1 )
              {
                v58 = v57 & 0xFFFFFFFFFFFFFFF8LL;
                v28 = v58;
                if ( (*(_BYTE *)(v58 + 44) & 4) == 0 )
                  break;
                v57 = *(_QWORD *)v58;
              }
            }
            v59 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v59 )
              BUG();
            v60 = *(_QWORD *)v59;
            v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v59 & 4) == 0 && (*(_BYTE *)(v59 + 44) & 4) != 0 )
            {
              while ( 1 )
              {
                v61 = v60 & 0xFFFFFFFFFFFFFFF8LL;
                v31 = v61;
                if ( (*(_BYTE *)(v61 + 44) & 4) == 0 )
                  break;
                v60 = *(_QWORD *)v61;
              }
            }
          }
          while ( v34 != v28 );
          if ( (*(_BYTE *)v28 & 4) != 0 )
            goto LABEL_102;
        }
        for ( ; (*(_BYTE *)(v28 + 44) & 8) != 0; v28 = *(_QWORD *)(v28 + 8) )
          ;
        goto LABEL_102;
      }
      v20 = *(_QWORD *)(a1 + 528);
      memset(v86, 0, 24);
      v21 = *a2;
      v22 = *(__int64 (**)())(*(_QWORD *)v20 + 984LL);
      if ( v22 != sub_2FDC730 )
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64 *, _QWORD))v22)(v20, v21, v86, 0) )
        {
          if ( v86[0] )
            j_j___libc_free_0(v86[0]);
          return 0;
        }
        v21 = *a2;
      }
      v23 = *(_DWORD *)(v21 + 44);
      if ( (v23 & 4) == 0 && (v23 & 8) != 0 )
        break;
      if ( (*(_BYTE *)(*(_QWORD *)(v21 + 16) + 25LL) & 4) == 0 )
        goto LABEL_18;
LABEL_20:
      if ( (*(_BYTE *)v21 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v21 + 44) & 8) != 0 )
          v21 = *(_QWORD *)(v21 + 8);
      }
      *a2 = *(_QWORD *)(v21 + 8);
      v24 = *a3;
      if ( !*a3 )
        BUG();
      if ( (*(_BYTE *)v24 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v24 + 44) & 8) != 0 )
          v24 = *(_QWORD *)(v24 + 8);
      }
      v25 = v86[0];
      *a3 = *(_QWORD *)(v24 + 8);
      if ( v25 )
        j_j___libc_free_0(v25);
      v12 = (_QWORD *)*a2;
      v11 = (_QWORD *)*a4;
      if ( *a4 == *a2 )
        return 1;
    }
    if ( !sub_2E88A90(v21, 1024, 1) )
LABEL_18:
      ++*a6;
    v21 = *a2;
    if ( !*a2 )
      BUG();
    goto LABEL_20;
  }
  return 1;
}

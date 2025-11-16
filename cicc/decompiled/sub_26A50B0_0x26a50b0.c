// Function: sub_26A50B0
// Address: 0x26a50b0
//
__int64 __fastcall sub_26A50B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r12d
  unsigned int v11; // r13d
  unsigned int v13; // eax
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rdi
  int v17; // r12d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rsi
  _BYTE *v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // r14
  int v29; // eax
  __int64 v30; // rdx
  int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  _QWORD *v37; // r15
  unsigned __int64 v38; // r13
  __int64 v39; // rdi
  __int64 v40; // rax
  unsigned int v41; // ecx
  _QWORD *v42; // rax
  _QWORD *j; // rdx
  __int64 v44; // r13
  int v45; // eax
  __int64 v46; // rdx
  unsigned int v47; // ecx
  _QWORD *v48; // rax
  _QWORD *k; // rdx
  unsigned int v50; // eax
  _QWORD *v51; // rdi
  _QWORD *v52; // rax
  unsigned int v53; // eax
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *i; // rdx
  unsigned int v57; // eax
  int v58; // r15d
  unsigned int v59; // eax
  int v60; // [rsp+4h] [rbp-6Ch]
  __int64 v61; // [rsp+8h] [rbp-68h]
  char v62; // [rsp+1Fh] [rbp-51h] BYREF
  __int64 v63; // [rsp+20h] [rbp-50h] BYREF
  char *v64; // [rsp+28h] [rbp-48h]
  char *v65; // [rsp+30h] [rbp-40h]

  if ( !(_BYTE)a2 )
  {
    LODWORD(v3) = a2;
    if ( (_BYTE)qword_4FF5148 )
    {
      sub_267B490(a1);
      if ( !byte_4FF5068 )
      {
LABEL_4:
        v4 = *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
        if ( !(_DWORD)v4 )
        {
LABEL_5:
          sub_267FFB0(*(__int64 **)(a1 + 72));
          v5 = *(_QWORD *)(a1 + 72);
          if ( *(_QWORD *)(v5 + 4752) )
          {
            v4 = *(_QWORD *)(a1 + 40);
            v62 = 0;
            v63 = a1;
            v64 = &v62;
            sub_26807A0(v5 + 4632, v4, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_267D2C0, (__int64)&v63);
            LOBYTE(v3) = v62 | v3;
          }
          if ( (_BYTE)qword_4FF4F88 )
          {
            v14 = *(_QWORD *)(a1 + 72);
            v62 = 0;
            v64 = (char *)a1;
            v65 = &v62;
            v63 = v14 + 26552;
            if ( !*(_BYTE *)(v14 + 34976)
              || (v15 = *(_QWORD *)(v14 + 26992)) != 0
              && !sub_B2FC80(v15)
              && (v16 = *(_QWORD *)(v14 + 27152)) != 0
              && !sub_B2FC80(v16) )
            {
              v4 = *(_QWORD *)(a1 + 40);
              sub_26807A0(v14 + 26552, v4, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_2677410, (__int64)&v63);
            }
            LOBYTE(v3) = v62 | v3;
          }
          v10 = sub_269B500(a1) | v3;
          if ( (_BYTE)qword_4FF53E8 )
          {
            v13 = sub_2680940(a1, v4, v6, v7, v8, v9);
            if ( (_BYTE)v13 )
            {
              v10 = v13;
              sub_269B500(a1);
            }
          }
          goto LABEL_9;
        }
LABEL_11:
        v4 = 0;
        sub_26A4B40((__int64 *)a1, 0);
        v60 = sub_2531E70(*(_QWORD *)(a1 + 80));
        if ( v60 )
        {
LABEL_12:
          LOBYTE(v3) = v60 == 0;
          goto LABEL_5;
        }
        v28 = **(_QWORD **)(*(_QWORD *)(a1 + 72) + 240LL);
        v29 = *(_DWORD *)(v28 + 80);
        ++*(_QWORD *)(v28 + 64);
        if ( v29 )
        {
          v41 = 4 * v29;
          v4 = 64;
          v30 = *(unsigned int *)(v28 + 88);
          if ( (unsigned int)(4 * v29) < 0x40 )
            v41 = 64;
          if ( (unsigned int)v30 > v41 )
          {
            v50 = v29 - 1;
            if ( v50 )
            {
              _BitScanReverse(&v50, v50);
              v51 = *(_QWORD **)(v28 + 72);
              v3 = (unsigned int)(1 << (33 - (v50 ^ 0x1F)));
              if ( (int)v3 < 64 )
                v3 = 64;
              if ( (_DWORD)v3 == (_DWORD)v30 )
              {
                *(_QWORD *)(v28 + 80) = 0;
                v52 = &v51[3 * v3];
                do
                {
                  if ( v51 )
                  {
                    *v51 = -4096;
                    v51[1] = -4096;
                  }
                  v51 += 3;
                }
                while ( v52 != v51 );
                goto LABEL_42;
              }
            }
            else
            {
              v51 = *(_QWORD **)(v28 + 72);
              LODWORD(v3) = 64;
            }
            v4 = 24 * v30;
            sub_C7D6A0((__int64)v51, 24 * v30, 8);
            v53 = sub_2671B90(v3);
            *(_DWORD *)(v28 + 88) = v53;
            if ( v53 )
            {
              v4 = 8;
              v54 = (_QWORD *)sub_C7D670(24LL * v53, 8);
              v55 = *(unsigned int *)(v28 + 88);
              *(_QWORD *)(v28 + 80) = 0;
              *(_QWORD *)(v28 + 72) = v54;
              for ( i = &v54[3 * v55]; i != v54; v54 += 3 )
              {
                if ( v54 )
                {
                  *v54 = -4096;
                  v54[1] = -4096;
                }
              }
              goto LABEL_42;
            }
            goto LABEL_40;
          }
        }
        else
        {
          if ( !*(_DWORD *)(v28 + 84) )
          {
LABEL_42:
            v31 = *(_DWORD *)(v28 + 48);
            ++*(_QWORD *)(v28 + 32);
            if ( v31 || *(_DWORD *)(v28 + 52) )
            {
              v32 = 4 * v31;
              v33 = *(unsigned int *)(v28 + 56);
              if ( v32 < 0x40 )
                v32 = 64;
              if ( (unsigned int)v33 > v32 )
              {
                sub_267DA90(v28 + 32);
              }
              else
              {
                v34 = *(_QWORD *)(v28 + 40);
                v35 = 32 * v33;
                v3 = v34 + 8;
                v61 = v34 + v35;
                if ( v34 != v34 + v35 )
                {
                  while ( 1 )
                  {
                    v36 = *(_QWORD *)(v3 - 8);
                    if ( v36 != -4096 )
                    {
                      if ( v36 != -8192 )
                      {
                        v37 = *(_QWORD **)v3;
                        while ( v37 != (_QWORD *)v3 )
                        {
                          v38 = (unsigned __int64)v37;
                          v37 = (_QWORD *)*v37;
                          v39 = *(_QWORD *)(v38 + 24);
                          if ( v39 )
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
                          v4 = 32;
                          j_j___libc_free_0(v38);
                        }
                      }
                      *(_QWORD *)(v3 - 8) = -4096;
                    }
                    v40 = v3 + 32;
                    v3 += 24;
                    if ( v61 == v3 )
                      break;
                    v3 = v40;
                  }
                }
                *(_QWORD *)(v28 + 48) = 0;
              }
            }
            goto LABEL_12;
          }
          v30 = *(unsigned int *)(v28 + 88);
          if ( (unsigned int)v30 > 0x40 )
          {
            v4 = 24 * v30;
            sub_C7D6A0(*(_QWORD *)(v28 + 72), 24 * v30, 8);
            *(_DWORD *)(v28 + 88) = 0;
LABEL_40:
            *(_QWORD *)(v28 + 72) = 0;
LABEL_41:
            *(_QWORD *)(v28 + 80) = 0;
            goto LABEL_42;
          }
        }
        v42 = *(_QWORD **)(v28 + 72);
        for ( j = &v42[3 * v30]; j != v42; *(v42 - 2) = -4096 )
        {
          *v42 = -4096;
          v42 += 3;
        }
        goto LABEL_41;
      }
    }
    else if ( !byte_4FF5068 )
    {
      goto LABEL_4;
    }
    sub_267C3E0(a1);
    v4 = *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
    if ( !(_DWORD)v4 )
      goto LABEL_5;
    goto LABEL_11;
  }
  v17 = 0;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 40) + 8LL) )
  {
    a2 = 1;
    sub_26A4B40((__int64 *)a1, 1);
    v17 = sub_2531E70(*(_QWORD *)(a1 + 80));
    if ( v17 )
    {
LABEL_35:
      LOBYTE(v17) = v17 == 0;
      goto LABEL_27;
    }
    v44 = **(_QWORD **)(*(_QWORD *)(a1 + 72) + 240LL);
    v45 = *(_DWORD *)(v44 + 80);
    ++*(_QWORD *)(v44 + 64);
    if ( !v45 )
    {
      if ( !*(_DWORD *)(v44 + 84) )
      {
LABEL_70:
        sub_BBC340(v44 + 32);
        goto LABEL_35;
      }
      v46 = *(unsigned int *)(v44 + 88);
      if ( (unsigned int)v46 > 0x40 )
      {
        a2 = 24 * v46;
        sub_C7D6A0(*(_QWORD *)(v44 + 72), 24 * v46, 8);
        *(_DWORD *)(v44 + 88) = 0;
LABEL_68:
        *(_QWORD *)(v44 + 72) = 0;
LABEL_69:
        *(_QWORD *)(v44 + 80) = 0;
        goto LABEL_70;
      }
LABEL_74:
      v48 = *(_QWORD **)(v44 + 72);
      for ( k = &v48[3 * v46]; k != v48; *(v48 - 2) = -4096 )
      {
        *v48 = -4096;
        v48 += 3;
      }
      goto LABEL_69;
    }
    v47 = 4 * v45;
    a2 = 64;
    v46 = *(unsigned int *)(v44 + 88);
    if ( (unsigned int)(4 * v45) < 0x40 )
      v47 = 64;
    if ( v47 >= (unsigned int)v46 )
      goto LABEL_74;
    v57 = v45 - 1;
    if ( v57 )
    {
      _BitScanReverse(&v57, v57);
      v58 = 1 << (33 - (v57 ^ 0x1F));
      if ( v58 < 64 )
        v58 = 64;
      if ( v58 == (_DWORD)v46 )
        goto LABEL_100;
    }
    else
    {
      v58 = 64;
    }
    a2 = 24 * v46;
    sub_C7D6A0(*(_QWORD *)(v44 + 72), 24 * v46, 8);
    v59 = sub_2671B90(v58);
    *(_DWORD *)(v44 + 88) = v59;
    if ( !v59 )
      goto LABEL_68;
    a2 = 8;
    *(_QWORD *)(v44 + 72) = sub_C7D670(24LL * v59, 8);
LABEL_100:
    sub_267E0E0(v44 + 64);
    goto LABEL_70;
  }
LABEL_27:
  sub_267FFB0(*(__int64 **)(a1 + 72));
  v10 = sub_2684FD0(a1, a2, v18, v19, v20, v21) | v17;
  v22 = sub_B6F970(**(_QWORD **)(a1 + 32));
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 32LL))(
         v22,
         "openmp-opt",
         10)
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 40LL))(
         v22,
         "openmp-opt",
         10)
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v22 + 24LL))(
         v22,
         "openmp-opt",
         10) )
  {
    v23 = *(_QWORD *)(a1 + 72);
    v24 = *(_QWORD *)(a1 + 40);
    v64 = (char *)a1;
    v63 = v23 + 32312;
    sub_26807A0(v23 + 32312, v24, (__int64 (__fastcall *)(__int64, _QWORD, __int64))sub_267D940, (__int64)&v63);
  }
LABEL_9:
  v11 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 72) + 34976LL);
  if ( (_BYTE)v11 )
  {
    v25 = sub_BA8CD0(*(_QWORD *)(a1 + 32), (__int64)"__llvm_rpc_client", 0x11u, 1);
    v26 = (__int64)v25;
    if ( v25 )
    {
      if ( !(unsigned int)sub_BD3960((__int64)v25) )
      {
        v10 = v11;
        v27 = sub_ACADE0(*(__int64 ***)(v26 + 8));
        sub_BD84D0(v26, v27);
        sub_B30290(v26);
      }
    }
  }
  return v10;
}

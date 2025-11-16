// Function: sub_28F38B0
// Address: 0x28f38b0
//
unsigned __int8 *__fastcall sub_28F38B0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v5; // rcx
  int v6; // eax
  unsigned __int8 *v7; // rbx
  unsigned __int8 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r14
  char v18; // r15
  int v19; // eax
  int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // rsi
  __int64 **v25; // r12
  _BYTE *v26; // rdi
  unsigned __int64 *v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned __int8 **v30; // rax
  unsigned __int8 *v31; // rdi
  unsigned __int8 v32; // al
  unsigned __int8 *v33; // rdi
  char v34; // al
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rdx
  void **v39; // rax
  __int64 v40; // rdx
  _BYTE *v41; // rsi
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  unsigned int v44; // r9d
  void **v45; // rax
  void **v46; // rsi
  char v47; // al
  unsigned int v48; // r9d
  void *v49; // rax
  _BYTE *v50; // rsi
  char v51; // [rsp+3h] [rbp-9Dh]
  int v52; // [rsp+4h] [rbp-9Ch]
  unsigned int v53; // [rsp+10h] [rbp-90h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  void **v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+20h] [rbp-80h]
  __int64 *v58; // [rsp+38h] [rbp-68h] BYREF
  const char *v59; // [rsp+40h] [rbp-60h] BYREF
  __int64 v60; // [rsp+48h] [rbp-58h]
  char *v61; // [rsp+50h] [rbp-50h]
  __int16 v62; // [rsp+60h] [rbp-40h]

  if ( *(_BYTE *)a1 <= 0x15u )
  {
    sub_B43CC0(a2);
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(unsigned __int8 *)(v5 + 8);
    if ( (unsigned int)(v6 - 17) <= 1 )
      LOBYTE(v6) = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
    v7 = (unsigned __int8 *)((unsigned __int8)v6 <= 3u || (_BYTE)v6 == 5 || (v6 & 0xFD) == 4
                           ? sub_96E680(12, a1)
                           : sub_AD6890(a1, 0));
    if ( v7 )
      return v7;
  }
  v9 = sub_28ED370((unsigned __int8 *)a1, 13, 14);
  v7 = v9;
  if ( v9 )
  {
    v10 = sub_28F38B0(*((_QWORD *)v9 - 8), a2, a3);
    if ( *((_QWORD *)v7 - 8) )
    {
      v11 = *((_QWORD *)v7 - 7);
      **((_QWORD **)v7 - 6) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *((_QWORD *)v7 - 6);
    }
    *((_QWORD *)v7 - 8) = v10;
    if ( v10 )
    {
      v12 = *(_QWORD *)(v10 + 16);
      *((_QWORD *)v7 - 7) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = v7 - 56;
      *((_QWORD *)v7 - 6) = v10 + 16;
      *(_QWORD *)(v10 + 16) = v7 - 64;
    }
    v13 = sub_28F38B0(*((_QWORD *)v7 - 4), a2, a3);
    if ( *((_QWORD *)v7 - 4) )
    {
      v14 = *((_QWORD *)v7 - 3);
      **((_QWORD **)v7 - 2) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *((_QWORD *)v7 - 2);
    }
    *((_QWORD *)v7 - 4) = v13;
    if ( v13 )
    {
      v15 = *(_QWORD *)(v13 + 16);
      *((_QWORD *)v7 - 3) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = v7 - 24;
      *((_QWORD *)v7 - 2) = v13 + 16;
      *(_QWORD *)(v13 + 16) = v7 - 32;
    }
    if ( *v7 == 42 )
    {
      sub_B447F0(v7, 0);
      sub_B44850(v7, 0);
    }
    sub_B444E0(v7, a2 + 24, 0);
    v62 = 773;
    v59 = sub_BD5D20((__int64)v7);
    v61 = ".neg";
    v60 = v16;
    sub_BD6B50(v7, &v59);
    sub_D68D20((__int64)&v59, 0, (__int64)v7);
    sub_28F19A0(a3, &v59);
    sub_D68D70(&v59);
    return v7;
  }
  v17 = *(_QWORD *)(a1 + 16);
  if ( !v17 )
  {
LABEL_36:
    v21 = (__int64 *)sub_BD5D20(a1);
    v62 = 773;
    v59 = (const char *)v21;
    v60 = v22;
    v61 = ".neg";
    v23 = sub_28E9340(a1, (__int64)&v59, a2 + 24, 0, 0, (_BYTE *)a2);
    v24 = *(__int64 **)(a2 + 48);
    v7 = (unsigned __int8 *)v23;
    v59 = (const char *)v24;
    if ( v24 )
    {
      v25 = (__int64 **)(v23 + 48);
      sub_B96E90((__int64)&v59, (__int64)v24, 1);
      if ( v25 == (__int64 **)&v59 )
      {
        if ( v59 )
          sub_B91220((__int64)&v59, (__int64)v59);
        goto LABEL_54;
      }
      v42 = *((_QWORD *)v7 + 6);
      if ( !v42 )
        goto LABEL_81;
    }
    else
    {
      v25 = (__int64 **)(v23 + 48);
      if ( (const char **)(v23 + 48) == &v59 )
        goto LABEL_54;
      v42 = *(_QWORD *)(v23 + 48);
      if ( !v42 )
        goto LABEL_54;
    }
    sub_B91220((__int64)v25, v42);
LABEL_81:
    v43 = (unsigned __int8 *)v59;
    *((_QWORD *)v7 + 6) = v59;
    if ( v43 )
      sub_B976B0((__int64)&v59, v43, (__int64)v25);
    goto LABEL_54;
  }
  while ( 1 )
  {
    v7 = *(unsigned __int8 **)(v17 + 24);
    v59 = 0;
    if ( *v7 == 44 )
    {
      if ( (unsigned __int8)sub_28EB290((__int64 **)&v59, (__int64)v7) )
        break;
    }
    v18 = sub_920620((__int64)v7);
    if ( !v18 )
      goto LABEL_35;
    v19 = *v7;
    if ( (unsigned __int8)v19 <= 0x1Cu )
      v20 = *((unsigned __int16 *)v7 + 1);
    else
      v20 = (unsigned __int8)v19 - 29;
    if ( v20 == 12 )
      goto LABEL_42;
    if ( v20 == 16 )
    {
      if ( (v7[1] & 0x10) != 0 )
      {
        v30 = (unsigned __int8 **)sub_986520((__int64)v7);
        v31 = *v30;
        v32 = **v30;
        if ( v32 == 18 )
        {
          v33 = *((void **)v31 + 3) == sub_C33340() ? (unsigned __int8 *)*((_QWORD *)v31 + 4) : v31 + 24;
          v34 = (v33[20] & 7) == 3;
        }
        else
        {
          v38 = *((_QWORD *)v31 + 1);
          v54 = v38;
          if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 > 1 || v32 > 0x15u )
            goto LABEL_35;
          v39 = (void **)sub_AD7630((__int64)v31, 0, v38);
          v40 = v54;
          if ( !v39 || (v55 = v39, *(_BYTE *)v39 != 18) )
          {
            if ( *(_BYTE *)(v40 + 8) == 17 )
            {
              v52 = *(_DWORD *)(v40 + 32);
              if ( v52 )
              {
                v51 = 0;
                v44 = 0;
                while ( 1 )
                {
                  v53 = v44;
                  v45 = (void **)sub_AD69F0(v31, v44);
                  v46 = v45;
                  if ( !v45 )
                    break;
                  v47 = *(_BYTE *)v45;
                  v48 = v53;
                  if ( v47 != 13 )
                  {
                    if ( v47 != 18 )
                      goto LABEL_35;
                    v49 = sub_C33340();
                    v48 = v53;
                    v50 = v46[3] == v49 ? v46[4] : v46 + 3;
                    if ( (v50[20] & 7) != 3 )
                      goto LABEL_35;
                    v51 = v18;
                  }
                  v44 = v48 + 1;
                  if ( v52 == v44 )
                  {
                    if ( v51 )
                      goto LABEL_41;
                    goto LABEL_35;
                  }
                }
              }
            }
            goto LABEL_35;
          }
          v41 = v39[3] == sub_C33340() ? v55[4] : v55 + 3;
          v34 = (v41[20] & 7) == 3;
        }
      }
      else
      {
        v58 = 0;
        v37 = (__int64 *)sub_986520((__int64)v7);
        v34 = sub_1008640(&v58, *v37);
      }
      if ( v34 )
        break;
    }
LABEL_35:
    v17 = *(_QWORD *)(v17 + 8);
    if ( !v17 )
      goto LABEL_36;
  }
LABEL_41:
  v19 = *v7;
LABEL_42:
  if ( (unsigned __int8)v19 <= 0x1Cu )
    BUG();
  if ( (unsigned int)(v19 - 42) <= 0x11 )
  {
    v26 = (_BYTE *)*((_QWORD *)v7 - 8);
    if ( *v26 <= 0x15u )
    {
      if ( (unsigned __int8)sub_AD6C40((__int64)v26) )
        goto LABEL_35;
    }
  }
  if ( *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) != *(_QWORD *)(*((_QWORD *)v7 + 5) + 72LL) )
    goto LABEL_35;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    sub_B445D0((__int64)&v59, (char *)a1);
    if ( (_BYTE)v61 )
    {
      v27 = (unsigned __int64 *)v59;
      v28 = (unsigned __int16)v60;
      v29 = *((_QWORD *)v7 + 5);
      if ( !v59 )
        BUG();
      goto LABEL_50;
    }
    goto LABEL_35;
  }
  v35 = *(_QWORD *)(sub_B43CB0((__int64)v7) + 80);
  if ( v35 )
    v35 -= 24;
  v36 = sub_AA5030(v35, 1);
  v29 = *((_QWORD *)v7 + 5);
  v28 = 0;
  if ( v36 )
    v36 -= 24;
  v27 = (unsigned __int64 *)(v36 + 24);
LABEL_50:
  if ( v27[2] != v29 )
  {
    v56 = v28;
    sub_AE8F80((char *)v7);
    v29 = v27[2];
    v28 = v56;
  }
  sub_B44550(v7, v29, v27, v28);
  if ( *v7 == 44 )
  {
    sub_B447F0(v7, 0);
    sub_B44850(v7, 0);
  }
  else
  {
    sub_B45560(v7, a2);
  }
LABEL_54:
  sub_D68D20((__int64)&v59, 0, (__int64)v7);
  sub_28F19A0(a3, &v59);
  sub_D68D70(&v59);
  return v7;
}

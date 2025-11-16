// Function: sub_A819E0
// Address: 0xa819e0
//
__int64 __fastcall sub_A819E0(__int64 a1, unsigned __int8 *a2, char a3)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  _BYTE *v10; // r15
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  _BYTE *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // r13
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  _BYTE *v17; // r9
  _BYTE *v18; // r10
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v21; // rax
  __int64 v22; // r15
  int v23; // edx
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // r15
  __int64 (__fastcall *v28)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v29; // rdi
  __int64 (__fastcall *v30)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  unsigned int *v31; // rax
  unsigned int *v32; // rax
  unsigned int *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r14
  int v37; // r14d
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned int *v42; // r13
  __int64 v43; // r14
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned int *v46; // r15
  __int64 v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // [rsp+0h] [rbp-B0h]
  __int64 v51; // [rsp+0h] [rbp-B0h]
  __int64 v52; // [rsp+8h] [rbp-A8h]
  _BYTE *v53; // [rsp+10h] [rbp-A0h]
  unsigned int *v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  _BYTE *v56; // [rsp+18h] [rbp-98h]
  _BYTE *v57; // [rsp+18h] [rbp-98h]
  unsigned int *v58; // [rsp+18h] [rbp-98h]
  unsigned int *v59; // [rsp+18h] [rbp-98h]
  __int64 v60; // [rsp+18h] [rbp-98h]
  _BYTE *v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+18h] [rbp-98h]
  _BYTE *v63; // [rsp+18h] [rbp-98h]
  _BYTE *v64; // [rsp+18h] [rbp-98h]
  unsigned int v65[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v66; // [rsp+40h] [rbp-70h]
  _BYTE v67[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v68; // [rsp+70h] [rbp-40h]

  v68 = 257;
  v6 = *((_QWORD *)a2 + 1);
  v7 = sub_A7EAA0(
         (unsigned int **)a1,
         0x31u,
         *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)],
         v6,
         (__int64)v67,
         0,
         v65[0],
         0);
  v68 = 257;
  v56 = (_BYTE *)v7;
  v53 = (_BYTE *)sub_A7EAA0(
                   (unsigned int **)a1,
                   0x31u,
                   *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
                   v6,
                   (__int64)v67,
                   0,
                   v65[0],
                   0);
  if ( a3 )
  {
    v8 = sub_AD64C0(v6, 32, 0);
    v9 = *(_QWORD *)(a1 + 80);
    v10 = (_BYTE *)v8;
    v66 = 257;
    v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v9 + 32LL);
    if ( v11 == sub_9201A0 )
    {
      if ( *v56 > 0x15u || *v10 > 0x15u )
      {
LABEL_39:
        v68 = 257;
        v12 = (_BYTE *)sub_B504D0(25, v56, v10, v67, 0, 0);
        (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v12,
          v65,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v31 = *(unsigned int **)a1;
        v50 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v50 )
        {
          do
          {
            v58 = v31;
            sub_B99FD0(v12, *v31, *((_QWORD *)v31 + 1));
            v31 = v58 + 4;
          }
          while ( (unsigned int *)v50 != v58 + 4 );
        }
LABEL_8:
        v68 = 257;
        v13 = sub_920F70((unsigned int **)a1, v12, v10, (__int64)v67, 0);
        v14 = *(_QWORD *)(a1 + 80);
        v15 = (_BYTE *)v13;
        v66 = 257;
        v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v14 + 32LL);
        if ( v16 == sub_9201A0 )
        {
          if ( *v53 > 0x15u || *v10 > 0x15u )
            goto LABEL_45;
          if ( (unsigned __int8)sub_AC47B0(25) )
            v17 = (_BYTE *)sub_AD5570(25, v53, v10, 0, 0);
          else
            v17 = (_BYTE *)sub_AABE40(25, v53, v10);
        }
        else
        {
          v17 = (_BYTE *)v16(v14, 25u, v53, v10, 0, 0);
        }
        if ( v17 )
        {
LABEL_14:
          v68 = 257;
          v18 = (_BYTE *)sub_920F70((unsigned int **)a1, v17, v10, (__int64)v67, 0);
          goto LABEL_15;
        }
LABEL_45:
        v68 = 257;
        v60 = sub_B504D0(25, v53, v10, v67, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v60,
          v65,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v33 = *(unsigned int **)a1;
        v17 = (_BYTE *)v60;
        v51 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v51 )
        {
          do
          {
            v54 = v33;
            v61 = v17;
            sub_B99FD0(v17, *v33, *((_QWORD *)v33 + 1));
            v17 = v61;
            v33 = v54 + 4;
          }
          while ( (unsigned int *)v51 != v54 + 4 );
        }
        goto LABEL_14;
      }
      if ( (unsigned __int8)sub_AC47B0(25) )
        v12 = (_BYTE *)sub_AD5570(25, v56, v10, 0, 0);
      else
        v12 = (_BYTE *)sub_AABE40(25, v56, v10);
    }
    else
    {
      v12 = (_BYTE *)v11(v9, 25u, v56, v10, 0, 0);
    }
    if ( v12 )
      goto LABEL_8;
    goto LABEL_39;
  }
  v25 = sub_AD64C0(v6, 0xFFFFFFFFLL, 0);
  v26 = *(_QWORD *)(a1 + 80);
  v66 = 257;
  v27 = (_BYTE *)v25;
  v28 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v26 + 16LL);
  if ( v28 != sub_9202E0 )
  {
    v15 = (_BYTE *)v28(v26, 28u, v56, v27);
    goto LABEL_31;
  }
  if ( *v56 <= 0x15u && *v27 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v15 = (_BYTE *)sub_AD5570(28, v56, v27, 0, 0);
    else
      v15 = (_BYTE *)sub_AABE40(28, v56, v27);
LABEL_31:
    if ( v15 )
      goto LABEL_32;
  }
  v68 = 257;
  v15 = (_BYTE *)sub_B504D0(28, v56, v27, v67, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v15,
    v65,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v32 = *(unsigned int **)a1;
  v52 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v52 )
  {
    do
    {
      v59 = v32;
      sub_B99FD0(v15, *v32, *((_QWORD *)v32 + 1));
      v32 = v59 + 4;
    }
    while ( (unsigned int *)v52 != v59 + 4 );
  }
LABEL_32:
  v29 = *(_QWORD *)(a1 + 80);
  v66 = 257;
  v30 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v29 + 16LL);
  if ( v30 != sub_9202E0 )
  {
    v18 = (_BYTE *)v30(v29, 28u, v53, v27);
    goto LABEL_37;
  }
  if ( *v53 <= 0x15u && *v27 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v18 = (_BYTE *)sub_AD5570(28, v53, v27, 0, 0);
    else
      v18 = (_BYTE *)sub_AABE40(28, v53, v27);
LABEL_37:
    if ( v18 )
      goto LABEL_15;
  }
  v68 = 257;
  v62 = sub_B504D0(28, v53, v27, v67, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v62,
    v65,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v18 = (_BYTE *)v62;
  v46 = *(unsigned int **)a1;
  v55 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v55 )
  {
    do
    {
      v47 = *((_QWORD *)v46 + 1);
      v48 = *v46;
      v46 += 4;
      v63 = v18;
      sub_B99FD0(v18, v48, v47);
      v18 = v63;
    }
    while ( (unsigned int *)v55 != v46 );
  }
LABEL_15:
  v19 = *(_QWORD *)(a1 + 80);
  v66 = 257;
  v20 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v19 + 32LL);
  if ( v20 != sub_9201A0 )
  {
    v64 = v18;
    v49 = v20(v19, 17u, v15, v18, 0, 0);
    v18 = v64;
    v22 = v49;
    goto LABEL_21;
  }
  if ( *v15 <= 0x15u && *v18 <= 0x15u )
  {
    v57 = v18;
    if ( (unsigned __int8)sub_AC47B0(17) )
      v21 = sub_AD5570(17, v15, v57, 0, 0);
    else
      v21 = sub_AABE40(17, v15, v57);
    v18 = v57;
    v22 = v21;
LABEL_21:
    if ( v22 )
      goto LABEL_22;
  }
  v68 = 257;
  v22 = sub_B504D0(17, v15, v18, v67, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v22,
    v65,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v42 = *(unsigned int **)a1;
  v43 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v43 )
  {
    do
    {
      v44 = *((_QWORD *)v42 + 1);
      v45 = *v42;
      v42 += 4;
      sub_B99FD0(v22, v45, v44);
    }
    while ( (unsigned int *)v43 != v42 );
  }
LABEL_22:
  v23 = *a2;
  if ( v23 == 40 )
  {
    v24 = 32LL * (unsigned int)sub_B491D0(a2);
  }
  else
  {
    v24 = 0;
    if ( v23 != 85 )
    {
      v24 = 64;
      if ( v23 != 34 )
LABEL_76:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_58;
  v34 = sub_BD2BC0(a2);
  v36 = v34 + v35;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v36 >> 4) )
      goto LABEL_76;
    goto LABEL_58;
  }
  if ( !(unsigned int)((v36 - sub_BD2BC0(a2)) >> 4) )
  {
LABEL_58:
    v40 = 0;
    goto LABEL_55;
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_76;
  v37 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v38 = sub_BD2BC0(a2);
  v40 = 32LL * (unsigned int)(*(_DWORD *)(v38 + v39 - 4) - v37);
LABEL_55:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v24 - v40) >> 5) == 4 )
    return sub_A7EE20(
             a1,
             *(_BYTE **)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
             v22,
             *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
  else
    return v22;
}

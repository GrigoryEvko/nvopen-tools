// Function: sub_17288D0
// Address: 0x17288d0
//
unsigned __int8 *__fastcall sub_17288D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        char a6,
        double a7,
        double a8,
        double a9)
{
  unsigned int v13; // edx
  __int64 v14; // r14
  bool v15; // al
  int v17; // r13d
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // r12
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // r13
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  unsigned __int8 *v41; // rsi
  __int64 *v42; // [rsp+10h] [rbp-B0h]
  char v43; // [rsp+18h] [rbp-A8h]
  unsigned int v44; // [rsp+20h] [rbp-A0h]
  __int64 v45; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v47; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v48[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v49; // [rsp+50h] [rbp-70h] BYREF
  const char *v50; // [rsp+58h] [rbp-68h]
  __int16 v51; // [rsp+60h] [rbp-60h]
  _QWORD *v52; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+78h] [rbp-48h]
  __int16 v54; // [rsp+80h] [rbp-40h]

  v13 = *(_DWORD *)(a3 + 8);
  v14 = *(_QWORD *)a2;
  if ( v13 > 0x40 )
  {
    v43 = a5;
    v44 = v13;
    v15 = sub_16A5220(a3, (const void **)a4);
    v13 = v44;
    a5 = v43;
    if ( v15 )
      goto LABEL_3;
LABEL_6:
    v17 = 35 - ((a6 == 0) - 1);
    if ( a5 )
    {
      v18 = *(_QWORD *)a3;
      v19 = v13 - 1;
      if ( v13 > 0x40 )
      {
        if ( (*(_QWORD *)(v18 + 8LL * (v19 >> 6)) & (1LL << v19)) == 0 || (unsigned int)sub_16A58A0(a3) != v19 )
          goto LABEL_10;
        goto LABEL_37;
      }
      if ( v18 == 1LL << v19 )
      {
LABEL_37:
        LOWORD(v17) = sub_15FF420(v17);
LABEL_28:
        v54 = 257;
        v30 = *(_QWORD *)(a1 + 8);
        v31 = sub_15A1070(v14, a4);
        if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v31 + 16) <= 0x10u )
        {
          v32 = sub_15A37B0(v17, (_QWORD *)a2, (_QWORD *)v31, 0);
          v28 = *(_QWORD *)(v30 + 96);
          v29 = v32;
LABEL_31:
          v33 = sub_14DBA30(v29, v28, 0);
          if ( v33 )
            return (unsigned __int8 *)v33;
          return (unsigned __int8 *)v29;
        }
        return sub_1727440(v30, v17, a2, v31, (__int64 *)&v52);
      }
    }
    else if ( v13 <= 0x40 )
    {
      if ( !*(_QWORD *)a3 )
        goto LABEL_28;
    }
    else if ( v13 == (unsigned int)sub_16A57B0(a3) )
    {
      goto LABEL_28;
    }
LABEL_10:
    v45 = *(_QWORD *)(a1 + 8);
    v48[0] = sub_1649960(a2);
    v51 = 773;
    v48[1] = v20;
    v49 = v48;
    v50 = ".off";
    v21 = sub_15A1070(v14, a3);
    if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v21 + 16) > 0x10u )
    {
      v54 = 257;
      v22 = sub_15FB440(13, (__int64 *)a2, v21, (__int64)&v52, 0);
      v34 = *(_QWORD *)(v45 + 8);
      if ( v34 )
      {
        v42 = *(__int64 **)(v45 + 16);
        sub_157E9D0(v34 + 40, v22);
        v35 = *v42;
        v36 = *(_QWORD *)(v22 + 24) & 7LL;
        *(_QWORD *)(v22 + 32) = v42;
        v35 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v22 + 24) = v35 | v36;
        *(_QWORD *)(v35 + 8) = v22 + 24;
        *v42 = *v42 & 7 | (v22 + 24);
      }
      sub_164B780(v22, (__int64 *)&v49);
      v47 = (unsigned __int8 *)v22;
      if ( !*(_QWORD *)(v45 + 80) )
        sub_4263D6(v22, &v49, v37);
      (*(void (__fastcall **)(__int64, unsigned __int8 **))(v45 + 88))(v45 + 64, &v47);
      v38 = *(_QWORD *)v45;
      if ( *(_QWORD *)v45 )
      {
        v47 = *(unsigned __int8 **)v45;
        sub_1623A60((__int64)&v47, v38, 2);
        v39 = *(_QWORD *)(v22 + 48);
        v40 = v22 + 48;
        if ( v39 )
        {
          sub_161E7C0(v22 + 48, v39);
          v40 = v22 + 48;
        }
        v41 = v47;
        *(_QWORD *)(v22 + 48) = v47;
        if ( v41 )
          sub_1623210((__int64)&v47, v41, v40);
      }
    }
    else
    {
      v22 = sub_15A2B60((__int64 *)a2, v21, 0, 0, a7, a8, a9);
      v23 = sub_14DBA30(v22, *(_QWORD *)(v45 + 96), 0);
      if ( v23 )
        v22 = v23;
    }
    LODWORD(v50) = *(_DWORD *)(a4 + 8);
    if ( (unsigned int)v50 > 0x40 )
      sub_16A4FD0((__int64)&v49, (const void **)a4);
    else
      v49 = *(_QWORD **)a4;
    sub_16A7590((__int64)&v49, (__int64 *)a3);
    v24 = (unsigned int)v50;
    LODWORD(v50) = 0;
    v53 = v24;
    v52 = v49;
    v25 = sub_15A1070(v14, (__int64)&v52);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( (unsigned int)v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
    v26 = *(_QWORD *)(a1 + 8);
    v54 = 257;
    if ( *(_BYTE *)(v22 + 16) <= 0x10u && *(_BYTE *)(v25 + 16) <= 0x10u )
    {
      v27 = sub_15A37B0(v17, (_QWORD *)v22, (_QWORD *)v25, 0);
      v28 = *(_QWORD *)(v26 + 96);
      v29 = v27;
      goto LABEL_31;
    }
    return sub_1727440(v26, v17, v22, v25, (__int64 *)&v52);
  }
  if ( *(_QWORD *)a3 != *(_QWORD *)a4 )
    goto LABEL_6;
LABEL_3:
  if ( a6 )
    return (unsigned __int8 *)sub_15A0640(v14);
  else
    return (unsigned __int8 *)sub_15A0600(v14);
}

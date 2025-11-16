// Function: sub_2528D80
// Address: 0x2528d80
//
__int64 __fastcall sub_2528D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  bool v10; // zf
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned int v14; // r13d
  __int64 *v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 *v19; // rax
  _QWORD *v20; // rdi
  char v21; // dl
  __int64 v22; // r12
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // [rsp-10h] [rbp-160h]
  __int64 v35; // [rsp+38h] [rbp-118h] BYREF
  char v36; // [rsp+47h] [rbp-109h] BYREF
  __int64 v37; // [rsp+48h] [rbp-108h] BYREF
  __int64 v38; // [rsp+50h] [rbp-100h] BYREF
  _QWORD **v39; // [rsp+58h] [rbp-F8h] BYREF
  __int64 *v40; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v41; // [rsp+68h] [rbp-E8h]
  __int64 *v42; // [rsp+70h] [rbp-E0h]
  __int64 *v43; // [rsp+78h] [rbp-D8h]
  _QWORD *v44; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+88h] [rbp-C8h]
  _QWORD v46[6]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+C0h] [rbp-90h] BYREF
  __int64 *v48; // [rsp+C8h] [rbp-88h]
  __int64 v49; // [rsp+D0h] [rbp-80h]
  int v50; // [rsp+D8h] [rbp-78h]
  unsigned __int8 v51; // [rsp+DCh] [rbp-74h]
  char v52; // [rsp+E0h] [rbp-70h] BYREF

  v10 = *(_QWORD *)(a7 + 16) == 0;
  v35 = a6;
  if ( v10 )
    goto LABEL_43;
  if ( a4 != sub_B43CB0(a2) )
  {
    v13 = sub_B43CB0(a2);
    if ( !*(_QWORD *)(a7 + 16) )
      sub_4263D6(a2, v13, v12);
    v14 = (*(__int64 (__fastcall **)(__int64, __int64))(a7 + 24))(a7, v13);
    if ( !(_BYTE)v14 )
    {
      if ( *(_BYTE *)(sub_251B1C0(*(_QWORD *)(a1 + 208), a4) + 114) )
      {
        v30 = *(_QWORD *)(a1 + 208);
        v31 = sub_B43CB0(a2);
        if ( *(_BYTE *)(sub_251B1C0(v30, v31) + 114) )
          return v14;
      }
    }
  }
  if ( !*(_QWORD *)(a7 + 16) )
  {
LABEL_43:
    if ( !v35 )
      return 1;
  }
  v15 = v46;
  v47 = 0;
  v16 = 1;
  v49 = 8;
  v50 = 0;
  v51 = 1;
  v44 = v46;
  v46[0] = a2;
  v48 = (__int64 *)&v52;
  v45 = 0x600000001LL;
  v17 = 1;
  while ( 1 )
  {
    v18 = v15[v17 - 1];
    LODWORD(v45) = v17 - 1;
    v37 = v18;
    if ( !(_BYTE)v16 )
      goto LABEL_14;
    v19 = v48;
    v16 = HIDWORD(v49);
    v15 = &v48[HIDWORD(v49)];
    if ( v48 != v15 )
    {
      while ( v18 != *v19 )
      {
        if ( v15 == ++v19 )
          goto LABEL_31;
      }
      goto LABEL_12;
    }
LABEL_31:
    if ( HIDWORD(v49) < (unsigned int)v49 )
    {
      ++HIDWORD(v49);
      *v15 = v18;
      ++v47;
    }
    else
    {
LABEL_14:
      sub_C8CC70((__int64)&v47, v18, (__int64)v15, v16, a5, a6);
      if ( !v21 )
        goto LABEL_12;
    }
    v22 = sub_B43CB0(v37);
    if ( a4 == v22 )
    {
      if ( !a3
        || (v41 = 0,
            v40 = (__int64 *)(a4 & 0xFFFFFFFFFFFFFFFCLL),
            v23 = a4 & 0xFFFFFFFFFFFFFFFCLL,
            nullsub_1518(),
            (v26 = sub_25285C0(a1, (__int64)v40, v41, a5, 1, 0, 1)) == 0)
        || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v26 + 112LL))(
             v26,
             a1,
             v37,
             a3,
             v35) )
      {
LABEL_25:
        v20 = v44;
        v14 = 1;
        goto LABEL_26;
      }
      if ( sub_B2FC80(a4) )
        goto LABEL_19;
    }
    else
    {
      if ( sub_B2FC80(a4) || !a3 )
      {
        v23 = v22 & 0xFFFFFFFFFFFFFFFCLL;
LABEL_19:
        v40 = (__int64 *)v23;
        v41 = 0;
        nullsub_1518();
        v24 = sub_25289A0(a1, (__int64)v40, v41, a5, 1, 0, 1);
        if ( !v24
          || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v24 + 112LL))(
               v24,
               a1,
               v37,
               a4,
               v35) )
        {
          goto LABEL_25;
        }
        goto LABEL_21;
      }
      v23 = a4 & 0xFFFFFFFFFFFFFFFCLL;
    }
    v40 = (__int64 *)v23;
    v41 = 0;
    nullsub_1518();
    v27 = sub_25285C0(a1, (__int64)v40, v41, a5, 1, 0, 1);
    v28 = *(_QWORD *)(a4 + 80);
    if ( !v28 )
      BUG();
    v29 = *(_QWORD *)(v28 + 32);
    if ( v29 )
      v29 -= 24;
    v23 = v22 & 0xFFFFFFFFFFFFFFFCLL;
    if ( !v27
      || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v27 + 112LL))(
           v27,
           a1,
           v29,
           a3,
           v35,
           v32) )
    {
      goto LABEL_19;
    }
LABEL_21:
    v40 = (__int64 *)v23;
    v41 = 0;
    nullsub_1518();
    v38 = sub_25285C0(a1, (__int64)v40, v41, a5, 1, 0, 1);
    v40 = &v38;
    v42 = &v37;
    v43 = &v35;
    v41 = a1;
    v36 = 0;
    LODWORD(v39) = 1;
    if ( !(unsigned __int8)sub_2526260(
                             a1,
                             (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2505E70,
                             (__int64)&v40,
                             v22,
                             a5,
                             &v36,
                             (int *)&v39,
                             1,
                             0,
                             0) )
    {
      if ( !*(_QWORD *)(a7 + 16) )
        goto LABEL_25;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(a7 + 24))(a7, v22) )
      {
        v39 = &v44;
        if ( !(unsigned __int8)sub_25230B0(
                                 a1,
                                 (__int64 (__fastcall *)(__int64, __int64 *))sub_25073A0,
                                 (__int64)&v39,
                                 v22,
                                 1,
                                 a5,
                                 &v36,
                                 0) )
          goto LABEL_25;
      }
    }
LABEL_12:
    v20 = v44;
    v17 = v45;
    v15 = v44;
    if ( !(_DWORD)v45 )
      break;
    v16 = v51;
  }
  v14 = 0;
LABEL_26:
  if ( v20 != v46 )
    _libc_free((unsigned __int64)v20);
  if ( !v51 )
    _libc_free((unsigned __int64)v48);
  return v14;
}

// Function: sub_1D1FED0
// Address: 0x1d1fed0
//
__int64 __fastcall sub_1D1FED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  unsigned int v10; // eax
  __int64 v11; // rsi
  _QWORD *v12; // rdx
  unsigned int v13; // eax
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r15
  unsigned int v21; // ebx
  __int64 v22; // rdx
  unsigned int v23; // eax
  const void *v24; // r13
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r12
  bool v27; // r14
  unsigned int v28; // ecx
  unsigned int v29; // r14d
  const void *v30; // [rsp+0h] [rbp-F0h]
  unsigned int v31; // [rsp+Ch] [rbp-E4h]
  int v32; // [rsp+10h] [rbp-E0h]
  bool v33; // [rsp+10h] [rbp-E0h]
  unsigned int v34; // [rsp+10h] [rbp-E0h]
  bool v36; // [rsp+2Fh] [rbp-C1h] BYREF
  _QWORD *v37; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-B8h]
  _QWORD *v39; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-A8h]
  _QWORD *v41; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-98h]
  _QWORD *v43; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-88h]
  const void *v45; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v46; // [rsp+78h] [rbp-78h]
  _QWORD *v47; // [rsp+80h] [rbp-70h] BYREF
  __int64 v48; // [rsp+88h] [rbp-68h]
  __int64 v49; // [rsp+90h] [rbp-60h]
  __int64 v50; // [rsp+98h] [rbp-58h]
  _QWORD *v51; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+A8h] [rbp-48h]
  __int64 v53; // [rsp+B0h] [rbp-40h]
  __int64 v54; // [rsp+B8h] [rbp-38h]

  v5 = 0;
  if ( sub_1D185B0(a4) )
    return v5;
  v47 = 0;
  v48 = 1;
  v49 = 0;
  v50 = 1;
  sub_1D1F820(a1, a4, a5, (unsigned __int64 *)&v47, 0);
  if ( (unsigned int)v48 <= 0x40 )
  {
    if ( !v47 )
      goto LABEL_5;
  }
  else
  {
    v32 = v48;
    if ( v32 == (unsigned int)sub_16A57B0((__int64)&v47) )
      goto LABEL_5;
  }
  v51 = 0;
  v52 = 1;
  v53 = 0;
  v54 = 1;
  sub_1D1F820(a1, a2, a3, (unsigned __int64 *)&v51, 0);
  v10 = v52;
  v38 = v52;
  if ( (unsigned int)v52 <= 0x40 )
  {
    v11 = (__int64)v51;
LABEL_18:
    v12 = (_QWORD *)(~v11 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v10));
    v37 = v12;
    goto LABEL_19;
  }
  sub_16A4FD0((__int64)&v37, (const void **)&v51);
  v10 = v38;
  if ( v38 <= 0x40 )
  {
    v11 = (__int64)v37;
    goto LABEL_18;
  }
  sub_16A8F40((__int64 *)&v37);
  v10 = v38;
  v12 = v37;
LABEL_19:
  v40 = v10;
  v13 = v48;
  v39 = v12;
  v38 = 0;
  v42 = v48;
  if ( (unsigned int)v48 <= 0x40 )
  {
    v14 = (unsigned __int64)v47;
LABEL_21:
    v15 = (_QWORD *)(~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13));
    v41 = v15;
    goto LABEL_22;
  }
  sub_16A4FD0((__int64)&v41, (const void **)&v47);
  v13 = v42;
  if ( v42 <= 0x40 )
  {
    v14 = (unsigned __int64)v41;
    goto LABEL_21;
  }
  sub_16A8F40((__int64 *)&v41);
  v15 = v41;
  v13 = v42;
LABEL_22:
  v43 = v15;
  v44 = v13;
  v42 = 0;
  sub_16A99B0((__int64)&v45, (__int64)&v39, (__int64 *)&v43, &v36);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  v16 = v54;
  if ( !v36 )
    goto LABEL_64;
  if ( (unsigned int)v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  if ( (unsigned int)v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
LABEL_5:
  if ( *(_WORD *)(a2 + 24) != 60 || (_DWORD)a3 != 1 )
    goto LABEL_6;
  v17 = v48;
  v31 = v48;
  v46 = v48;
  if ( (unsigned int)v48 <= 0x40 )
  {
    v18 = (unsigned __int64)v47;
LABEL_47:
    v30 = (const void *)(~v18 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31));
    v45 = v30;
    goto LABEL_48;
  }
  sub_16A4FD0((__int64)&v45, (const void **)&v47);
  v31 = v46;
  if ( v46 <= 0x40 )
  {
    v18 = (unsigned __int64)v45;
    v17 = v48;
    goto LABEL_47;
  }
  sub_16A8F40((__int64 *)&v45);
  v31 = v46;
  v30 = v45;
  v17 = v48;
LABEL_48:
  v46 = 0;
  v42 = v17;
  LODWORD(v52) = v31;
  v51 = v30;
  if ( v17 <= 0x40 )
  {
    v19 = (unsigned __int64)v47;
LABEL_50:
    v20 = ~v19 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
LABEL_51:
    v33 = (v20 & 1) == (_QWORD)v30;
    goto LABEL_52;
  }
  sub_16A4FD0((__int64)&v41, (const void **)&v47);
  LOBYTE(v17) = v42;
  if ( v42 <= 0x40 )
  {
    v19 = (unsigned __int64)v41;
    goto LABEL_50;
  }
  sub_16A8F40((__int64 *)&v41);
  v28 = v42;
  v20 = (unsigned __int64)v41;
  v42 = 0;
  if ( v28 <= 0x40 )
    goto LABEL_51;
  *v41 &= 1uLL;
  v34 = v28;
  memset((void *)(v20 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v28 + 63) >> 6) - 8);
  v43 = (_QWORD *)v20;
  v44 = v34;
  v33 = sub_16A5220((__int64)&v43, (const void **)&v51);
  j_j___libc_free_0_0(v20);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
LABEL_52:
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  v5 = 0;
  if ( v33 )
    goto LABEL_7;
LABEL_6:
  v5 = 1;
  if ( *(_WORD *)(a4 + 24) != 60 )
    goto LABEL_7;
  v5 = a5;
  if ( (_DWORD)a5 == 1 )
  {
    v51 = 0;
    v52 = 1;
    v53 = 0;
    v54 = 1;
    sub_1D1F820(a1, a2, a3, (unsigned __int64 *)&v51, 0);
    v21 = v52;
    v44 = v52;
    if ( (unsigned int)v52 > 0x40 )
    {
      sub_16A4FD0((__int64)&v43, (const void **)&v51);
      v21 = v44;
      if ( v44 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v43);
        v21 = v44;
        v24 = v43;
        v23 = v52;
LABEL_74:
        v46 = v21;
        v45 = v24;
        v44 = 0;
        v40 = v23;
        if ( v23 > 0x40 )
        {
          sub_16A4FD0((__int64)&v39, (const void **)&v51);
          LOBYTE(v23) = v40;
          if ( v40 > 0x40 )
          {
            sub_16A8F40((__int64 *)&v39);
            v29 = v40;
            v26 = (unsigned __int64)v39;
            v40 = 0;
            if ( v29 > 0x40 )
            {
              *v39 &= 1uLL;
              memset((void *)(v26 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v29 + 63) >> 6) - 8);
              v42 = v29;
              v41 = (_QWORD *)v26;
              v27 = sub_16A5220((__int64)&v41, &v45);
              j_j___libc_free_0_0(v26);
              if ( v40 > 0x40 && v39 )
                j_j___libc_free_0_0(v39);
LABEL_78:
              if ( v21 > 0x40 && v24 )
                j_j___libc_free_0_0(v24);
              if ( v44 > 0x40 && v43 )
                j_j___libc_free_0_0(v43);
              v16 = v54;
              if ( !v27 )
              {
                if ( (unsigned int)v54 > 0x40 && v53 )
                  j_j___libc_free_0_0(v53);
                if ( (unsigned int)v52 > 0x40 && v51 )
                  j_j___libc_free_0_0(v51);
                goto LABEL_7;
              }
LABEL_64:
              if ( v16 > 0x40 && v53 )
                j_j___libc_free_0_0(v53);
              if ( (unsigned int)v52 > 0x40 && v51 )
                j_j___libc_free_0_0(v51);
              v5 = 0;
              goto LABEL_7;
            }
LABEL_77:
            v27 = (v26 & 1) == (_QWORD)v24;
            goto LABEL_78;
          }
          v25 = (unsigned __int64)v39;
        }
        else
        {
          v25 = (unsigned __int64)v51;
        }
        v26 = ~v25 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v23);
        goto LABEL_77;
      }
      v22 = (__int64)v43;
      v23 = v52;
    }
    else
    {
      v22 = (__int64)v51;
      v23 = v52;
    }
    v24 = (const void *)(~v22 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21));
    v43 = v24;
    goto LABEL_74;
  }
  v5 = 1;
LABEL_7:
  if ( (unsigned int)v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( (unsigned int)v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  return v5;
}

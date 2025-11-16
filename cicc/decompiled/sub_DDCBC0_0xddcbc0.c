// Function: sub_DDCBC0
// Address: 0xddcbc0
//
__int64 __fastcall sub_DDCBC0(__int64 *a1, int a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *(__fastcall *v8)(__int64, __int64, __int64, unsigned int); // r14
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  _QWORD *v12; // rax
  unsigned int v13; // r8d
  int v14; // r10d
  unsigned int v15; // r15d
  __int64 v16; // rsi
  unsigned int v17; // ebx
  __int64 v18; // rax
  unsigned int v19; // ebx
  _QWORD *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rsi
  _BYTE *v23; // rcx
  _BYTE *v24; // rdx
  __int64 v26; // r14
  unsigned int v27; // edx
  bool v28; // al
  bool v29; // r14
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rdx
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // esi
  unsigned int v38; // ebx
  unsigned int v39; // eax
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned int v43; // eax
  _QWORD *v45; // [rsp+8h] [rbp-A8h]
  _QWORD *v46; // [rsp+10h] [rbp-A0h]
  unsigned int v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+20h] [rbp-90h]
  __int64 v51; // [rsp+20h] [rbp-90h]
  void *v52; // [rsp+28h] [rbp-88h]
  unsigned int v53; // [rsp+28h] [rbp-88h]
  unsigned __int64 v54; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v55; // [rsp+38h] [rbp-78h]
  unsigned __int64 v56; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v57; // [rsp+48h] [rbp-68h]
  unsigned __int64 v58; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v59; // [rsp+58h] [rbp-58h]
  unsigned __int64 v60; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v61; // [rsp+68h] [rbp-48h]
  unsigned __int64 v62; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v63; // [rsp+78h] [rbp-38h]

  switch ( a2 )
  {
    case 15:
      v52 = sub_DCC810;
      break;
    case 17:
      v52 = sub_DCA690;
      break;
    case 13:
      v52 = sub_DC7ED0;
      if ( !a3 )
        goto LABEL_5;
      goto LABEL_35;
    default:
      BUG();
  }
  if ( !a3 )
  {
LABEL_5:
    v8 = sub_DC2B70;
    goto LABEL_6;
  }
LABEL_35:
  v8 = sub_DC5000;
LABEL_6:
  v9 = sub_D95540(a4);
  v10 = sub_BCCE00(*(_QWORD **)v9, 2 * (*(_DWORD *)(v9 + 8) >> 8));
  v11 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _QWORD))v52)(a1, a4, a5, 0, 0);
  v46 = v8((__int64)a1, v11, v10, 0);
  v45 = v8((__int64)a1, a4, v10, 0);
  v12 = v8((__int64)a1, a5, v10, 0);
  if ( v46 == (_QWORD *)((__int64 (__fastcall *)(__int64 *, _QWORD *, _QWORD *, _QWORD, _QWORD))v52)(a1, v45, v12, 0, 0) )
    return 1;
  v14 = a2;
  LOBYTE(v13) = a2 == 17;
  v15 = v13;
  LOBYTE(v15) = a6 == 0 || a2 == 17;
  if ( (_BYTE)v15 )
    return 0;
  if ( !*(_WORD *)(a5 + 24) )
  {
    v16 = *(_QWORD *)(a5 + 32);
    v17 = *(_DWORD *)(v16 + 32);
    v55 = v17;
    if ( v17 <= 0x40 )
    {
      v54 = *(_QWORD *)(v16 + 24);
      if ( !a3 )
      {
        v57 = v17;
LABEL_12:
        v18 = v54;
LABEL_13:
        v56 = v18;
LABEL_14:
        if ( !a3 )
        {
          if ( v14 == 15 )
          {
            v59 = v17;
            if ( v17 <= 0x40 )
            {
              v58 = 0;
              v63 = v17;
              v19 = 37;
              goto LABEL_18;
            }
            v19 = 37;
            sub_C43690((__int64)&v58, 0, 0);
            v43 = v59;
LABEL_80:
            v63 = v43;
            if ( v43 > 0x40 )
            {
              sub_C43780((__int64)&v62, (const void **)&v58);
              goto LABEL_19;
            }
LABEL_18:
            v62 = v58;
LABEL_19:
            sub_C45EE0((__int64)&v62, (__int64 *)&v56);
            v61 = v63;
            v60 = v62;
            v20 = sub_DA26C0(a1, (__int64)&v60);
            v21 = a6;
            v22 = v19;
            v23 = (_BYTE *)a4;
            v24 = v20;
            goto LABEL_20;
          }
          v37 = v17;
          v38 = 37;
          sub_9691E0((__int64)&v58, v37, -1, 1u, 0);
          v39 = v59;
LABEL_70:
          v63 = v39;
          if ( v39 > 0x40 )
          {
            sub_C43780((__int64)&v62, (const void **)&v58);
            goto LABEL_72;
          }
LABEL_71:
          v62 = v58;
LABEL_72:
          sub_C46B40((__int64)&v62, (__int64 *)&v56);
          v61 = v63;
          v60 = v62;
          v40 = sub_DA26C0(a1, (__int64)&v60);
          v21 = a6;
          v22 = v38;
          v24 = (_BYTE *)a4;
          v23 = v40;
LABEL_20:
          v15 = sub_DDCB50(a1, v22, v24, v23, v21);
          if ( v61 > 0x40 && v60 )
            j_j___libc_free_0_0(v60);
          if ( v59 > 0x40 && v58 )
            j_j___libc_free_0_0(v58);
LABEL_26:
          if ( v57 > 0x40 && v56 )
            j_j___libc_free_0_0(v56);
          if ( v55 > 0x40 && v54 )
            j_j___libc_free_0_0(v54);
          return v15;
        }
        v53 = v17 - 1;
LABEL_65:
        v36 = 1LL << v53;
        if ( (a2 == 15) != (_BYTE)v15 )
        {
          v59 = v17;
          if ( v17 > 0x40 )
          {
            sub_C43690((__int64)&v58, 0, 0);
            v17 = v59;
            v36 = 1LL << v53;
            if ( v59 > 0x40 )
            {
              *(_QWORD *)(v58 + 8LL * (v53 >> 6)) |= 1LL << v53;
              v43 = v59;
              v19 = 41;
              goto LABEL_80;
            }
          }
          else
          {
            v58 = 0;
          }
          v63 = v17;
          v19 = 41;
          v58 |= v36;
          goto LABEL_18;
        }
        v59 = v17;
        v41 = ~v36;
        if ( v17 > 0x40 )
        {
          v51 = v41;
          sub_C43690((__int64)&v58, -1, 1);
          v17 = v59;
          v41 = v51;
          if ( v59 > 0x40 )
          {
            *(_QWORD *)(v58 + 8LL * (v53 >> 6)) &= v51;
            v38 = 41;
            v39 = v59;
            goto LABEL_70;
          }
        }
        else
        {
          v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
          if ( !v17 )
            v42 = 0;
          v58 = v42;
        }
        v63 = v17;
        v38 = 41;
        v58 &= v41;
        goto LABEL_71;
      }
      v53 = v17 - 1;
      v26 = 1LL << ((unsigned __int8)v17 - 1);
      goto LABEL_39;
    }
    sub_C43780((__int64)&v54, (const void **)(v16 + 24));
    v17 = v55;
    v14 = a2;
    if ( a3 )
    {
      v53 = v55 - 1;
      v26 = 1LL << ((unsigned __int8)v55 - 1);
      if ( v55 <= 0x40 )
      {
LABEL_39:
        v18 = v54;
        if ( (v26 & v54) == 0 )
        {
          v57 = v17;
          goto LABEL_13;
        }
        v57 = v17;
        v27 = v17;
        v56 = v54;
        v63 = v17;
        v62 = 0;
        goto LABEL_42;
      }
      v35 = v53 >> 6;
      if ( (*(_QWORD *)(v54 + 8 * v35) & v26) != 0 )
      {
        v50 = 8 * v35;
        v57 = v55;
        sub_C43780((__int64)&v56, (const void **)&v54);
        v63 = v17;
        sub_C43690((__int64)&v62, 0, 0);
        v27 = v63;
        if ( v63 > 0x40 )
        {
          *(_QWORD *)(v62 + v50) |= v26;
          v27 = v63;
LABEL_43:
          if ( v55 <= 0x40 )
          {
            v29 = v54 == v62;
          }
          else
          {
            v49 = v27;
            v28 = sub_C43C50((__int64)&v54, (const void **)&v62);
            v27 = v49;
            v29 = v28;
          }
          if ( v27 > 0x40 && v62 )
            j_j___libc_free_0_0(v62);
          if ( v29 )
            goto LABEL_26;
          v30 = v55;
          v61 = v55;
          if ( v55 > 0x40 )
          {
            sub_C43780((__int64)&v60, (const void **)&v54);
            v30 = v61;
            if ( v61 > 0x40 )
            {
              sub_C43D10((__int64)&v60);
LABEL_54:
              sub_C46250((__int64)&v60);
              v33 = v61;
              v34 = v60;
              v61 = 0;
              v63 = v33;
              v62 = v60;
              if ( v57 > 0x40 && v56 )
              {
                j_j___libc_free_0_0(v56);
                v34 = v62;
                v33 = v63;
              }
              v57 = v33;
              LOBYTE(v15) = 1;
              v56 = v34;
              v63 = 0;
              sub_969240((__int64 *)&v62);
              sub_969240((__int64 *)&v60);
              goto LABEL_65;
            }
            v31 = v60;
          }
          else
          {
            v31 = v54;
          }
          v32 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v30) & ~v31;
          if ( !v30 )
            v32 = 0;
          v60 = v32;
          goto LABEL_54;
        }
LABEL_42:
        v62 |= v26;
        goto LABEL_43;
      }
      v57 = v55;
    }
    else
    {
      v57 = v55;
      if ( v55 <= 0x40 )
        goto LABEL_12;
    }
    sub_C43780((__int64)&v56, (const void **)&v54);
    v14 = a2;
    goto LABEL_14;
  }
  return v15;
}

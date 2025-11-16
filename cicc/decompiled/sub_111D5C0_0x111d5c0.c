// Function: sub_111D5C0
// Address: 0x111d5c0
//
__int64 __fastcall sub_111D5C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rax
  char v13; // al
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r15
  int v17; // eax
  __int64 v18; // r13
  unsigned int v19; // eax
  unsigned int v20; // r14d
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // r15
  _BYTE *v24; // rax
  unsigned int v25; // eax
  unsigned int *v26; // rax
  unsigned int v27; // [rsp+Ch] [rbp-124h]
  __int64 v28; // [rsp+10h] [rbp-120h]
  __int64 v29; // [rsp+10h] [rbp-120h]
  __int64 v30; // [rsp+18h] [rbp-118h]
  unsigned int *v31; // [rsp+18h] [rbp-118h]
  __int64 v32; // [rsp+30h] [rbp-100h] BYREF
  __int64 v33; // [rsp+38h] [rbp-F8h] BYREF
  const void **v34; // [rsp+40h] [rbp-F0h] BYREF
  int v35; // [rsp+48h] [rbp-E8h] BYREF
  char v36; // [rsp+4Ch] [rbp-E4h]
  unsigned __int64 v37; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v38; // [rsp+58h] [rbp-D8h]
  __int64 v39; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v41; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v42; // [rsp+78h] [rbp-B8h]
  unsigned __int64 v43; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v44; // [rsp+88h] [rbp-A8h]
  char v45[32]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v46; // [rsp+B0h] [rbp-80h]
  __int64 v47; // [rsp+C0h] [rbp-70h] BYREF
  __int64 *v48; // [rsp+C8h] [rbp-68h]
  __int64 *v49; // [rsp+D0h] [rbp-60h] BYREF
  char v50; // [rsp+D8h] [rbp-58h]
  const void ***v51; // [rsp+E0h] [rbp-50h] BYREF
  char v52; // [rsp+E8h] [rbp-48h]
  __int64 *v53; // [rsp+F0h] [rbp-40h]

  v2 = *(_QWORD *)(a1 - 64);
  v47 = (__int64)&v35;
  v49 = &v33;
  v48 = &v32;
  v50 = 0;
  v51 = &v34;
  v52 = 0;
  v53 = &v32;
  v3 = *(_QWORD *)(v2 + 16);
  v35 = 42;
  v36 = 0;
  if ( v3
    && !*(_QWORD *)(v3 + 8)
    && *(_BYTE *)v2 == 56
    && (v12 = *(_BYTE **)(v2 - 64), *v12 == 54)
    && *((_QWORD *)v12 - 8)
    && (v32 = *((_QWORD *)v12 - 8), (unsigned __int8)sub_991580((__int64)&v49, *((_QWORD *)v12 - 4))) )
  {
    v13 = sub_991580((__int64)&v51, *(_QWORD *)(v2 - 32));
    v4 = *(_QWORD *)(a1 - 32);
    if ( v13 && v4 == *v53 )
    {
      if ( v47 )
      {
        v14 = sub_B53900(a1);
        v15 = v47;
        *(_DWORD *)v47 = v14;
        *(_BYTE *)(v15 + 4) = BYTE4(v14);
      }
      goto LABEL_24;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 - 32);
  }
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( *(_BYTE *)v4 != 56 )
    return 0;
  v8 = *(_BYTE **)(v4 - 64);
  if ( *v8 != 54 )
    return 0;
  v9 = *((_QWORD *)v8 - 8);
  if ( !v9 )
    return 0;
  *v48 = v9;
  if ( !(unsigned __int8)sub_991580((__int64)&v49, *((_QWORD *)v8 - 4))
    || !(unsigned __int8)sub_991580((__int64)&v51, *(_QWORD *)(v4 - 32))
    || *(_QWORD *)(a1 - 64) != *v53 )
  {
    return 0;
  }
  if ( v47 )
  {
    v10 = sub_B53960(a1);
    v11 = v47;
    *(_DWORD *)v47 = v10;
    *(_BYTE *)(v11 + 4) = BYTE4(v10);
  }
LABEL_24:
  v16 = (__int64 *)v33;
  if ( *(_DWORD *)(v33 + 8) <= 0x40u )
  {
    if ( *(const void **)v33 == *v34 )
    {
      v17 = v35;
      if ( v35 != 32 )
        goto LABEL_27;
LABEL_56:
      v27 = 36;
      goto LABEL_29;
    }
    return 0;
  }
  if ( !sub_C43C50(v33, v34) )
    return 0;
  v17 = v35;
  if ( v35 == 32 )
    goto LABEL_56;
LABEL_27:
  v6 = 0;
  if ( v17 != 33 )
    return v6;
  v27 = 35;
LABEL_29:
  v18 = *(_QWORD *)(v32 + 8);
  v19 = sub_BCB060(v18);
  v38 = v19;
  v20 = v19;
  if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v37, v19, 0);
    LODWORD(v48) = v38;
    if ( v38 > 0x40 )
    {
      sub_C43780((__int64)&v47, (const void **)&v37);
      sub_C46B40((__int64)&v47, v16);
      v25 = (unsigned int)v48;
      LODWORD(v48) = v20;
      v40 = v25;
      v39 = v47;
      goto LABEL_61;
    }
  }
  else
  {
    v37 = v19;
    LODWORD(v48) = v19;
  }
  v47 = v37;
  sub_C46B40((__int64)&v47, v16);
  v21 = (unsigned int)v48;
  LODWORD(v48) = v20;
  v40 = v21;
  v39 = v47;
  if ( v20 <= 0x40 )
  {
    v47 = 1;
    v42 = v20;
    goto LABEL_33;
  }
LABEL_61:
  sub_C43690((__int64)&v47, 1, 0);
  v42 = (unsigned int)v48;
  if ( (unsigned int)v48 > 0x40 )
  {
    sub_C43780((__int64)&v41, (const void **)&v47);
    goto LABEL_34;
  }
LABEL_33:
  v41 = v47;
LABEL_34:
  sub_C47AC0((__int64)&v41, (__int64)&v39);
  if ( (unsigned int)v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  v22 = v42;
  v44 = v42;
  if ( v42 <= 0x40 )
  {
    v43 = v41;
    goto LABEL_39;
  }
  sub_C43780((__int64)&v43, (const void **)&v41);
  v22 = v44;
  if ( v44 <= 0x40 )
  {
LABEL_39:
    if ( v22 == 1 )
      v43 = 0;
    else
      v43 >>= 1;
    goto LABEL_41;
  }
  sub_C482E0((__int64)&v43, 1u);
LABEL_41:
  v46 = 257;
  v28 = sub_AD8D80(v18, (__int64)&v43);
  v30 = v32;
  v23 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a2 + 80) + 32LL))(
          *(_QWORD *)(a2 + 80),
          13,
          v32,
          v28,
          0,
          0);
  if ( !v23 )
  {
    LOWORD(v51) = 257;
    v23 = sub_B504D0(13, v30, v28, (__int64)&v47, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v23,
      v45,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v26 = *(unsigned int **)a2;
    v29 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v29 )
    {
      do
      {
        v31 = v26;
        sub_B99FD0(v23, *v26, *((_QWORD *)v26 + 1));
        v26 = v31 + 4;
      }
      while ( (unsigned int *)v29 != v31 + 4 );
    }
  }
  LOWORD(v51) = 257;
  v24 = (_BYTE *)sub_AD8D80(v18, (__int64)&v41);
  v6 = sub_92B530((unsigned int **)a2, v27, v23, v24, (__int64)&v47);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  return v6;
}

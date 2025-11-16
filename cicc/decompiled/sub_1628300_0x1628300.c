// Function: sub_1628300
// Address: 0x1628300
//
__int64 __fastcall sub_1628300(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // r11
  __int64 v5; // rcx
  signed int v6; // r13d
  signed int v7; // r14d
  __int64 v8; // r15
  int v9; // eax
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 *v18; // r15
  __int64 *v19; // r12
  _QWORD *v20; // r8
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 *v23; // rdi
  __int64 v24; // r12
  unsigned int v26; // r13d
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 *i; // rcx
  __int64 v32; // r15
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  signed int v40; // [rsp+18h] [rbp-D8h]
  signed int v41; // [rsp+1Ch] [rbp-D4h]
  __int64 v42; // [rsp+20h] [rbp-D0h]
  __int64 v43; // [rsp+28h] [rbp-C8h]
  int v44; // [rsp+30h] [rbp-C0h]
  __int64 v45; // [rsp+38h] [rbp-B8h]
  _QWORD *v46; // [rsp+38h] [rbp-B8h]
  __int64 v47; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-A8h]
  __int64 v49; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v50; // [rsp+58h] [rbp-98h]
  __int64 *v51; // [rsp+60h] [rbp-90h] BYREF
  __int64 v52; // [rsp+68h] [rbp-88h]
  _BYTE v53[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v54; // [rsp+90h] [rbp-60h] BYREF
  __int64 v55; // [rsp+98h] [rbp-58h]
  __int64 v56; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+A8h] [rbp-48h]

  if ( !a1 )
    return 0;
  v2 = a2;
  if ( !a2 )
    return 0;
  if ( a1 == a2 )
    return a1;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(unsigned int *)(a2 + 8);
  v6 = 0;
  v7 = 0;
  v51 = (__int64 *)v53;
  v52 = 0x400000000LL;
  v40 = (unsigned int)v5 >> 1;
  v41 = (unsigned int)v4 >> 1;
  if ( (unsigned int)v4 >> 1 && (unsigned int)v5 >> 1 )
  {
    while ( 1 )
    {
      v45 = v5;
      v44 = 2 * v7;
      v42 = v4;
      v43 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (2 * v7 - v4)) + 136LL);
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (2 * v6 - v5)) + 136LL);
      v9 = sub_16AEA10(v43 + 24, v8 + 24);
      v10 = 2 * v6;
      if ( v9 >= 0 )
      {
        ++v6;
        sub_161E3A0((__int64)&v51, v8, *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v10 + 1 - v45)) + 136LL));
        if ( v41 <= v7 )
          goto LABEL_12;
      }
      else
      {
        ++v7;
        sub_161E3A0((__int64)&v51, v43, *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v44 + 1 - v42)) + 136LL));
        if ( v41 <= v7 )
          goto LABEL_12;
      }
      if ( v40 <= v6 )
        break;
      v5 = *(unsigned int *)(a2 + 8);
      v4 = *(unsigned int *)(a1 + 8);
    }
  }
  if ( v41 > v7 )
  {
    v32 = 2 * ((unsigned int)(v41 - 1 - v7) + (__int64)v7) + 3;
    v33 = 2 * v7 + 1LL;
    do
    {
      v34 = *(unsigned int *)(a1 + 8);
      v35 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v33 - v34)) + 136LL);
      v36 = v33 - 1;
      v33 += 2;
      sub_161E3A0((__int64)&v51, *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v36 - v34)) + 136LL), v35);
    }
    while ( v32 != v33 );
    v2 = a2;
  }
LABEL_12:
  if ( v40 > v6 )
  {
    v11 = 2 * v6 + 1LL;
    v12 = 2 * ((unsigned int)(v40 - 1 - v6) + (__int64)v6) + 3;
    do
    {
      v13 = *(unsigned int *)(v2 + 8);
      v14 = *(_QWORD *)(*(_QWORD *)(v2 + 8 * (v11 - v13)) + 136LL);
      v15 = v11 - 1;
      v11 += 2;
      sub_161E3A0((__int64)&v51, *(_QWORD *)(*(_QWORD *)(v2 + 8 * (v15 - v13)) + 136LL), v14);
    }
    while ( v12 != v11 );
  }
  v16 = (unsigned int)v52;
  if ( (unsigned int)v52 > 4 )
  {
    if ( !(unsigned __int8)sub_161DF70((__int64 *)&v51, *v51, v51[1]) )
    {
      LODWORD(v16) = v52;
      if ( (_DWORD)v52 != 2 )
        goto LABEL_17;
      goto LABEL_47;
    }
    v26 = v16 - 2;
    v27 = 16;
    v28 = 8 * v16;
    do
    {
      v51[(unsigned __int64)v27 / 8 - 2] = v51[(unsigned __int64)v27 / 8];
      v27 += 8;
    }
    while ( v27 != v28 );
    v29 = (unsigned int)v52;
    v17 = v26;
    LODWORD(v16) = v52;
    if ( v26 >= (unsigned __int64)(unsigned int)v52 )
    {
      if ( v26 <= (unsigned __int64)(unsigned int)v52 )
        goto LABEL_16;
      if ( v26 > (unsigned __int64)HIDWORD(v52) )
      {
        sub_16CD150(&v51, v53, v26, 8);
        v29 = (unsigned int)v52;
        v17 = v26;
      }
      v30 = &v51[v29];
      for ( i = &v51[v17]; i != v30; ++v30 )
      {
        if ( v30 )
          *v30 = 0;
      }
    }
    LODWORD(v52) = v26;
    LODWORD(v16) = v26;
    goto LABEL_18;
  }
LABEL_16:
  if ( (_DWORD)v16 != 2 )
  {
LABEL_17:
    v17 = (unsigned int)v16;
    goto LABEL_18;
  }
LABEL_47:
  v37 = (__int64)v51;
  v38 = v51[1];
  v50 = *(_DWORD *)(v38 + 32);
  if ( v50 > 0x40 )
  {
    sub_16A4FD0(&v49, v38 + 24);
    v37 = (__int64)v51;
  }
  else
  {
    v49 = *(_QWORD *)(v38 + 24);
  }
  v39 = *(_QWORD *)v37;
  v48 = *(_DWORD *)(*(_QWORD *)v37 + 32LL);
  if ( v48 > 0x40 )
    sub_16A4FD0(&v47, v39 + 24);
  else
    v47 = *(_QWORD *)(v39 + 24);
  sub_15898E0((__int64)&v54, (__int64)&v47, &v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( sub_158A0B0((__int64)&v54) )
  {
    if ( v57 > 0x40 && v56 )
      j_j___libc_free_0_0(v56);
    if ( (unsigned int)v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    v24 = 0;
    goto LABEL_28;
  }
  if ( v57 > 0x40 && v56 )
    j_j___libc_free_0_0(v56);
  if ( (unsigned int)v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  v17 = (unsigned int)v52;
  LODWORD(v16) = v52;
LABEL_18:
  v54 = &v56;
  v55 = 0x400000000LL;
  if ( (unsigned int)v16 > 4 )
  {
    sub_16CD150(&v54, &v56, v17, 8);
    v17 = (unsigned int)v52;
  }
  v18 = v51;
  v19 = &v51[v17];
  if ( v51 == v19 )
  {
    v22 = (unsigned int)v55;
  }
  else
  {
    do
    {
      v20 = sub_1624210(*v18);
      v21 = (unsigned int)v55;
      if ( (unsigned int)v55 >= HIDWORD(v55) )
      {
        v46 = v20;
        sub_16CD150(&v54, &v56, 0, 8);
        v21 = (unsigned int)v55;
        v20 = v46;
      }
      ++v18;
      v54[v21] = (__int64)v20;
      v22 = (unsigned int)(v55 + 1);
      LODWORD(v55) = v55 + 1;
    }
    while ( v19 != v18 );
  }
  v23 = (__int64 *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    v23 = (__int64 *)*v23;
  v24 = sub_1627350(v23, v54, (__int64 *)v22, 0, 1);
  if ( v54 != &v56 )
    _libc_free((unsigned __int64)v54);
LABEL_28:
  if ( v51 != (__int64 *)v53 )
    _libc_free((unsigned __int64)v51);
  return v24;
}

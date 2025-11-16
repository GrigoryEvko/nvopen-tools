// Function: sub_2A85BE0
// Address: 0x2a85be0
//
__int64 __fastcall sub_2A85BE0(_QWORD *a1, unsigned __int8 a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  unsigned __int64 v5; // rax
  __int64 *v6; // rsi
  unsigned __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  const char *v18; // r8
  char v19; // di
  int v20; // ecx
  __int64 v21; // r10
  char v22; // al
  unsigned int v23; // esi
  unsigned __int16 v24; // dx
  unsigned int v25; // ebx
  _QWORD *v26; // rdi
  __int64 *v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // rcx
  int v30; // eax
  int v31; // eax
  unsigned int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  _QWORD *v36; // rdi
  _QWORD *v37; // rdi
  __int64 v38; // rsi
  unsigned __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned __int16 v42; // bx
  __int64 v43; // r15
  _QWORD *v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v48; // [rsp+8h] [rbp-98h]
  __int64 *i; // [rsp+8h] [rbp-98h]
  __int64 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  const char *v52; // [rsp+10h] [rbp-90h]
  __int64 v53; // [rsp+10h] [rbp-90h]
  unsigned __int16 v54; // [rsp+18h] [rbp-88h]
  unsigned __int16 v55; // [rsp+18h] [rbp-88h]
  const char *v56; // [rsp+18h] [rbp-88h]
  __int64 *v57; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v58; // [rsp+28h] [rbp-78h]
  __int64 *v59; // [rsp+30h] [rbp-70h]
  const char *v60; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int16 v61; // [rsp+48h] [rbp-58h]
  char v62; // [rsp+60h] [rbp-40h]
  char v63; // [rsp+61h] [rbp-3Fh]

  v2 = a1 + 9;
  v3 = (_QWORD *)a1[10];
  v57 = 0;
  v58 = 0;
  v59 = 0;
  if ( v3 == a1 + 9 )
    return 0;
  do
  {
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      v5 = v3[3] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v5 == v3 + 3 )
        goto LABEL_73;
      if ( !v5 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
LABEL_73:
        BUG();
      if ( *(_BYTE *)(v5 - 24) != 30 )
        goto LABEL_3;
      v6 = v58;
      v60 = (const char *)(v3 - 3);
      if ( v58 != v59 )
        break;
      sub_F38A10((__int64)&v57, v58, &v60);
LABEL_3:
      v3 = (_QWORD *)v3[1];
      if ( v2 == v3 )
        goto LABEL_13;
    }
    if ( v58 )
    {
      *v58 = (__int64)(v3 - 3);
      v6 = v58;
    }
    v58 = v6 + 1;
    v3 = (_QWORD *)v3[1];
  }
  while ( v2 != v3 );
LABEL_13:
  v7 = (unsigned __int64)v57;
  if ( v57 == v58 )
    goto LABEL_16;
  if ( (char *)v58 - (char *)v57 != 8 )
  {
    v63 = 1;
    v60 = "UnifiedReturnBlock";
    v62 = 3;
    v10 = sub_B2BE50((__int64)a1);
    v11 = sub_22077B0(0x50u);
    v12 = v11;
    if ( v11 )
      sub_AA4D50(v11, v10, (__int64)&v60, (__int64)a1, 0);
    v13 = a1[3];
    if ( *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL) == 7 )
    {
      sub_B43C20((__int64)&v60, v12);
      v41 = sub_B2BE50((__int64)a1);
      v42 = v61;
      v43 = v41;
      v56 = v60;
      v44 = sub_BD2C40(72, 0);
      if ( v44 )
        sub_B4BB80((__int64)v44, v43, 0, 0, (__int64)v56, v42);
      v16 = 0;
    }
    else
    {
      v63 = 1;
      v60 = "UnifiedRetVal";
      v62 = 3;
      v14 = v58 - v57;
      v50 = **(_QWORD **)(v13 + 16);
      v15 = sub_BD2DA0(80);
      v16 = v15;
      if ( v15 )
      {
        sub_B44260(v15, v50, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v16 + 72) = v14;
        sub_BD6B50((unsigned __int8 *)v16, &v60);
        sub_BD2A10(v16, *(_DWORD *)(v16 + 72), 1);
        sub_B44240((_QWORD *)v16, v12, (unsigned __int64 *)(v12 + 48), 0);
        sub_B43C20((__int64)&v60, v12);
        v17 = sub_B2BE50((__int64)a1);
        v18 = v60;
        v19 = v61;
        v20 = 1;
        v21 = v17;
        v22 = HIBYTE(v61);
        v23 = 1;
      }
      else
      {
        sub_B44240(0, v12, (unsigned __int64 *)(v12 + 48), 0);
        sub_B43C20((__int64)&v60, v12);
        v46 = sub_B2BE50((__int64)a1);
        v18 = v60;
        v19 = v61;
        v20 = 0;
        v21 = v46;
        v23 = 0;
        v22 = HIBYTE(v61);
      }
      v48 = v21;
      LOBYTE(v24) = v19;
      v51 = (__int64)v18;
      v25 = v20 & 0x1FFFFFFF;
      HIBYTE(v24) = v22;
      v54 = v24;
      v26 = sub_BD2C40(72, v23);
      if ( v26 )
        sub_B4BB80((__int64)v26, v48, v16, v25, v51, v54);
    }
    v27 = v57;
    for ( i = v58; i != v27; ++v27 )
    {
      v38 = *v27;
      if ( v16 )
      {
        v39 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v39 == v38 + 48 )
          goto LABEL_70;
        if ( !v39 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30 > 0xA )
LABEL_70:
          BUG();
        if ( (*(_BYTE *)(v39 - 17) & 0x40) != 0 )
          v28 = *(__int64 **)(v39 - 32);
        else
          v28 = (__int64 *)(v39 - 24 - 32LL * (*(_DWORD *)(v39 - 20) & 0x7FFFFFF));
        v29 = *v28;
        v30 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
        if ( v30 == *(_DWORD *)(v16 + 72) )
        {
          v53 = v29;
          sub_B48D90(v16);
          v29 = v53;
          v30 = *(_DWORD *)(v16 + 4) & 0x7FFFFFF;
        }
        v31 = (v30 + 1) & 0x7FFFFFF;
        v32 = v31 | *(_DWORD *)(v16 + 4) & 0xF8000000;
        v33 = *(_QWORD *)(v16 - 8) + 32LL * (unsigned int)(v31 - 1);
        *(_DWORD *)(v16 + 4) = v32;
        if ( *(_QWORD *)v33 )
        {
          v34 = *(_QWORD *)(v33 + 8);
          **(_QWORD **)(v33 + 16) = v34;
          if ( v34 )
            *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
        }
        *(_QWORD *)v33 = v29;
        if ( v29 )
        {
          v35 = *(_QWORD *)(v29 + 16);
          *(_QWORD *)(v33 + 8) = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = v33 + 8;
          *(_QWORD *)(v33 + 16) = v29 + 16;
          *(_QWORD *)(v29 + 16) = v33;
        }
        *(_QWORD *)(*(_QWORD *)(v16 - 8)
                  + 32LL * *(unsigned int *)(v16 + 72)
                  + 8LL * ((*(_DWORD *)(v16 + 4) & 0x7FFFFFFu) - 1)) = v38;
      }
      v36 = (_QWORD *)((*(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24);
      if ( (*(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        v36 = 0;
      sub_B43D60(v36);
      sub_B43C20((__int64)&v60, v38);
      v52 = v60;
      v55 = v61;
      v37 = sub_BD2C40(72, 1u);
      if ( v37 )
        sub_B4C8F0((__int64)v37, v12, 1u, (__int64)v52, v55);
    }
    if ( !a2 )
      goto LABEL_52;
    v40 = (a1[9] & 0xFFFFFFFFFFFFFFF8LL) - 24;
    if ( (a1[9] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v40 = 0;
    if ( v12 == v40 )
    {
LABEL_52:
      v7 = (unsigned __int64)v57;
      v8 = 1;
      goto LABEL_17;
    }
    v45 = v12;
LABEL_67:
    sub_AA4AF0(v45, v40);
    v7 = (unsigned __int64)v57;
    v8 = a2;
LABEL_17:
    if ( v7 )
      goto LABEL_18;
    return v8;
  }
  if ( !a2 )
  {
LABEL_16:
    v8 = 0;
    goto LABEL_17;
  }
  v40 = (a1[9] & 0xFFFFFFFFFFFFFFF8LL) - 24;
  if ( (a1[9] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v40 = 0;
  if ( *v57 != v40 )
  {
    v45 = *v57;
    goto LABEL_67;
  }
  v8 = 0;
LABEL_18:
  j_j___libc_free_0(v7);
  return v8;
}

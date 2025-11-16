// Function: sub_2AE2E80
// Address: 0x2ae2e80
//
void __fastcall sub_2AE2E80(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 *v4; // r13
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rax
  char v17; // di
  __int64 v18; // r14
  __int64 *v19; // r13
  int v20; // ebx
  __int64 v21; // rsi
  __int64 *v22; // rdi
  __int64 *v23; // rsi
  int v24; // eax
  __int64 v25; // rdx
  int v26; // r11d
  unsigned int v27; // r8d
  bool v28; // r9
  bool v29; // zf
  int v30; // r12d
  bool v31; // r9
  __int64 *v32; // rbx
  __int64 v33; // rsi
  char v34; // al
  __int64 v35; // rsi
  __int64 *v36; // r12
  __int64 *v37; // rax
  __int64 v38; // r15
  char v39; // al
  __int64 *v40; // rsi
  unsigned int v41; // eax
  unsigned int v42; // r8d
  __int64 *v43; // rax
  __int64 v44; // rax
  int v45; // ecx
  __int64 v46; // rsi
  int v47; // ecx
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rsi
  unsigned int v52; // esi
  __int64 v53; // rax
  int v54; // eax
  int v55; // r8d
  unsigned int v56; // r8d
  __int64 v57; // [rsp+8h] [rbp-128h]
  __int64 v58; // [rsp+18h] [rbp-118h]
  int v59; // [rsp+30h] [rbp-100h]
  unsigned int v60; // [rsp+34h] [rbp-FCh]
  __int64 v61; // [rsp+38h] [rbp-F8h]
  __int64 v63; // [rsp+48h] [rbp-E8h]
  __int64 v64; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v65; // [rsp+58h] [rbp-D8h] BYREF
  __int64 *v66; // [rsp+60h] [rbp-D0h] BYREF
  __int64 *v67; // [rsp+68h] [rbp-C8h] BYREF
  __int64 *v68; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+78h] [rbp-B8h]
  _BYTE v70[48]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v72; // [rsp+B8h] [rbp-78h]
  __int64 *v73; // [rsp+C0h] [rbp-70h] BYREF
  unsigned int v74; // [rsp+C8h] [rbp-68h]
  char v75; // [rsp+100h] [rbp-30h] BYREF

  v2 = a1;
  if ( LOBYTE(qword_500D340[17]) )
    sub_2AD7E50(a1, a2);
  v3 = *(_QWORD *)(a1 + 8);
  v68 = (__int64 *)v70;
  v69 = 0x600000000LL;
  sub_D472F0(v3, (__int64)&v68);
  v4 = v68;
  v5 = &v68[(unsigned int)v69];
  if ( v68 != v5 )
  {
    do
    {
      v6 = sub_AA5930(*v4);
      v8 = v7;
      v9 = v6;
      while ( v8 != v9 )
      {
        sub_DACA20(*(_QWORD *)(*(_QWORD *)(v2 + 16) + 112LL), *(_QWORD *)(v2 + 8), v9);
        if ( !v9 )
          BUG();
        v10 = *(_QWORD *)(v9 + 32);
        if ( !v10 )
          BUG();
        v9 = 0;
        if ( *(_BYTE *)(v10 - 24) == 84 )
          v9 = v10 - 24;
      }
      ++v4;
    }
    while ( v5 != v4 );
  }
  sub_DAC210(*(_QWORD *)(*(_QWORD *)(v2 + 16) + 112LL), *(_QWORD *)(v2 + 8));
  sub_D9D700(*(_QWORD *)(*(_QWORD *)(v2 + 16) + 112LL), 0);
  if ( sub_2BF3F10(*(_QWORD *)(a2 + 920)) )
  {
    v11 = *(__int64 **)(v2 + 312);
    v12 = &v11[*(unsigned int *)(v2 + 320)];
    while ( v12 != v11 )
    {
      v13 = *v11++;
      sub_2AE2140(v2, v13);
    }
    v14 = sub_2BF3F10(*(_QWORD *)(a2 + 920));
    v64 = sub_2BF04D0(v14);
    v15 = *sub_2ACA740(a2 + 120, &v64);
    v71 = 0;
    v72 = 1;
    v58 = v15;
    v16 = (__int64 *)&v73;
    do
    {
      *v16 = -4096;
      v16 += 2;
    }
    while ( v16 != (__int64 *)&v75 );
    v17 = v72;
    v63 = v58 + 48;
    if ( v58 + 48 != *(_QWORD *)(v58 + 56) )
    {
      v57 = v2;
      v18 = *(_QWORD *)(v58 + 56);
      while ( 1 )
      {
        while ( 1 )
        {
          v38 = v18;
          v18 = *(_QWORD *)(v18 + 8);
          v39 = *(_BYTE *)(v38 - 24);
          v36 = (__int64 *)(v38 - 24);
          if ( (unsigned __int8)(v39 - 90) > 2u && v39 != 63 )
            goto LABEL_29;
          if ( (v17 & 1) == 0 )
            break;
          v19 = (__int64 *)&v73;
          v20 = 3;
LABEL_19:
          v21 = 32LL * (*(_DWORD *)(v38 - 20) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v38 - 17) & 0x40) != 0 )
          {
            v22 = *(__int64 **)(v38 - 32);
            v23 = &v22[(unsigned __int64)v21 / 8];
          }
          else
          {
            v22 = &v36[v21 / 0xFFFFFFFFFFFFFFF8LL];
            v23 = (__int64 *)(v38 - 24);
          }
          v67 = (__int64 *)sub_2ABF340(v22, v23);
          LODWORD(v66) = *(unsigned __int8 *)(v38 - 24) - 29;
          v24 = sub_C4ECF0((int *)&v66, (__int64 *)&v67);
          v25 = v38 - 24;
          v26 = 1;
          v27 = v20 & v24;
          v28 = v36 + 512 == 0;
          v29 = v36 + 1024 == 0;
          v30 = v20;
          v31 = v29 || v28;
          while ( 1 )
          {
            v32 = &v19[2 * v27];
            v33 = *v32;
            if ( *v32 == -4096 || *v32 == -8192 || v31 )
              break;
            v59 = v26;
            v60 = v27;
            v61 = v25;
            v34 = sub_B46220(v25, v33);
            v25 = v61;
            v27 = v60;
            v26 = v59;
            v31 = 0;
            if ( v34 )
              goto LABEL_25;
LABEL_66:
            v56 = v26 + v27;
            ++v26;
            v27 = v30 & v56;
          }
          if ( v25 != v33 )
          {
            if ( v33 == -4096 )
            {
              v36 = (__int64 *)v25;
              goto LABEL_26;
            }
            goto LABEL_66;
          }
LABEL_25:
          v35 = v32[1];
          v36 = (__int64 *)v25;
          if ( !v35 )
            goto LABEL_26;
          sub_BD84D0(v25, v35);
          sub_B43D60(v36);
          v17 = v72;
          if ( v63 == v18 )
          {
LABEL_46:
            v2 = v57;
            goto LABEL_47;
          }
        }
        v19 = v73;
        if ( v74 )
        {
          v20 = v74 - 1;
          goto LABEL_19;
        }
LABEL_26:
        v65 = v36;
        if ( !(unsigned __int8)sub_2ABF6D0((__int64)&v71, &v65, &v66) )
          break;
        v37 = v66 + 1;
LABEL_28:
        *v37 = (__int64)v36;
        v17 = v72;
LABEL_29:
        if ( v63 == v18 )
          goto LABEL_46;
      }
      v40 = v66;
      ++v71;
      v67 = v66;
      v41 = ((unsigned int)v72 >> 1) + 1;
      if ( (v72 & 1) != 0 )
      {
        v42 = 4;
        if ( 4 * v41 < 0xC )
          goto LABEL_41;
      }
      else
      {
        v42 = v74;
        if ( 4 * v41 < 3 * v74 )
        {
LABEL_41:
          if ( v42 - (v41 + HIDWORD(v72)) > v42 >> 3 )
          {
LABEL_42:
            LODWORD(v72) = v72 & 1 | (2 * v41);
            if ( *v40 != -4096 )
              --HIDWORD(v72);
            v43 = v65;
            v40[1] = 0;
            *v40 = (__int64)v43;
            v37 = v40 + 1;
            goto LABEL_28;
          }
          v52 = v42;
LABEL_61:
          sub_2ABF860((__int64)&v71, v52);
          sub_2ABF6D0((__int64)&v71, &v65, &v67);
          v40 = v67;
          v41 = ((unsigned int)v72 >> 1) + 1;
          goto LABEL_42;
        }
      }
      v52 = 2 * v42;
      goto LABEL_61;
    }
LABEL_47:
    if ( (v17 & 1) != 0 )
    {
      v44 = *(_QWORD *)(v2 + 24);
      v45 = *(_DWORD *)(v44 + 24);
      v46 = *(_QWORD *)(v44 + 8);
      if ( !v45 )
        goto LABEL_59;
    }
    else
    {
      sub_C7D6A0((__int64)v73, 16LL * v74, 8);
      v53 = *(_QWORD *)(v2 + 24);
      v45 = *(_DWORD *)(v53 + 24);
      v46 = *(_QWORD *)(v53 + 8);
      if ( !v45 )
        goto LABEL_59;
    }
    v47 = v45 - 1;
    v48 = v47 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
    v49 = (__int64 *)(v46 + 16LL * v48);
    v50 = *v49;
    if ( v58 == *v49 )
    {
LABEL_50:
      v51 = v49[1];
LABEL_51:
      sub_F70890(
        *(_QWORD *)(v2 + 8),
        v51,
        *(_QWORD *)(v2 + 8),
        (unsigned int)(*(_DWORD *)(v2 + 72) * *(_DWORD *)(v2 + 88)));
      goto LABEL_52;
    }
    v54 = 1;
    while ( v50 != -4096 )
    {
      v55 = v54 + 1;
      v48 = v47 & (v54 + v48);
      v49 = (__int64 *)(v46 + 16LL * v48);
      v50 = *v49;
      if ( v58 == *v49 )
        goto LABEL_50;
      v54 = v55;
    }
LABEL_59:
    v51 = 0;
    goto LABEL_51;
  }
LABEL_52:
  if ( v68 != (__int64 *)v70 )
    _libc_free((unsigned __int64)v68);
}

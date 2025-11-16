// Function: sub_14AD470
// Address: 0x14ad470
//
void __fastcall sub_14AD470(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v8; // rdi
  __int64 v9; // rax
  char v10; // dl
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 *v15; // rax
  __int64 *v16; // rsi
  __int64 *v17; // rcx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  char v22; // cl
  int v23; // esi
  __int64 v24; // rdi
  __int64 v25; // r8
  int v26; // esi
  unsigned int v27; // r12d
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 *v30; // r10
  __int64 v31; // r10
  __int64 v32; // rax
  __int64 *v33; // r12
  unsigned __int64 v34; // r15
  __int64 *v35; // r13
  __int64 v36; // rbx
  __int64 v37; // r10
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 *v40; // r11
  __int64 v41; // r12
  __int64 *v42; // r9
  __int64 v43; // rdx
  __int64 v44; // r11
  int v45; // r10d
  int v46; // r11d
  __int64 v47; // r10
  int v48; // edx
  int v49; // r9d
  int v50; // r11d
  int v51; // r10d
  int v52; // r9d
  __int64 v53; // r11
  int v54; // r11d
  __int64 v55; // r9
  __int64 v56; // [rsp+18h] [rbp-E8h]
  _QWORD *v57; // [rsp+20h] [rbp-E0h]
  int v58; // [rsp+20h] [rbp-E0h]
  unsigned int v59; // [rsp+28h] [rbp-D8h]
  unsigned int v60; // [rsp+2Ch] [rbp-D4h]
  unsigned int v61; // [rsp+2Ch] [rbp-D4h]
  unsigned int v62; // [rsp+2Ch] [rbp-D4h]
  __int64 v63; // [rsp+30h] [rbp-D0h]
  _QWORD *v65; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+58h] [rbp-A8h]
  _QWORD v67[4]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v68; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v69; // [rsp+88h] [rbp-78h]
  __int64 *v70; // [rsp+90h] [rbp-70h]
  __int64 v71; // [rsp+98h] [rbp-68h]
  int v72; // [rsp+A0h] [rbp-60h]
  _BYTE v73[88]; // [rsp+A8h] [rbp-58h] BYREF

  v66 = 0x400000001LL;
  v69 = (__int64 *)v73;
  v70 = (__int64 *)v73;
  v63 = a2 + 16;
  v65 = v67;
  v67[0] = a1;
  v8 = v67;
  LODWORD(v9) = 1;
  v68 = 0;
  v71 = 4;
  v72 = 0;
  do
  {
    while ( 1 )
    {
      v13 = v8[(unsigned int)v9 - 1];
      LODWORD(v66) = v9 - 1;
      v14 = sub_14AD280(v13, a3, a5);
      v15 = v69;
      if ( v70 != v69 )
        goto LABEL_2;
      v16 = &v69[HIDWORD(v71)];
      if ( v69 != v16 )
      {
        v17 = 0;
        while ( v14 != *v15 )
        {
          if ( *v15 == -2 )
            v17 = v15;
          if ( v16 == ++v15 )
          {
            if ( !v17 )
              goto LABEL_43;
            *v17 = v14;
            --v72;
            ++v68;
            goto LABEL_3;
          }
        }
        goto LABEL_9;
      }
LABEL_43:
      if ( HIDWORD(v71) < (unsigned int)v71 )
      {
        ++HIDWORD(v71);
        *v16 = v14;
        ++v68;
      }
      else
      {
LABEL_2:
        sub_16CCBA0(&v68, v14);
        if ( !v10 )
          goto LABEL_9;
      }
LABEL_3:
      v11 = *(_BYTE *)(v14 + 16);
      if ( v11 <= 0x17u )
        goto LABEL_6;
      if ( v11 == 79 )
        break;
      if ( v11 != 77 )
      {
LABEL_6:
        v12 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, v63, 0, 8);
          v12 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v14;
        ++*(_DWORD *)(a2 + 8);
        goto LABEL_9;
      }
      v21 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
      v22 = *(_BYTE *)(v14 + 23) & 0x40;
      if ( !a4 )
        goto LABEL_35;
      v23 = *(_DWORD *)(a4 + 24);
      if ( !v23 )
        goto LABEL_35;
      v24 = *(_QWORD *)(v14 + 40);
      v25 = *(_QWORD *)(a4 + 8);
      v26 = v23 - 1;
      v27 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v28 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v28;
      v30 = v28;
      if ( v24 != *v28 )
      {
        v62 = v26 & (((unsigned int)*(_QWORD *)(v14 + 40) >> 9) ^ ((unsigned int)v24 >> 4));
        v44 = *v28;
        v45 = 1;
        while ( v44 != -8 )
        {
          v46 = v45 + 1;
          v47 = v26 & (v62 + v45);
          v58 = v46;
          v62 = v47;
          v30 = (__int64 *)(v25 + 16 * v47);
          v44 = *v30;
          if ( v24 == *v30 )
            goto LABEL_33;
          v45 = v58;
        }
        goto LABEL_35;
      }
LABEL_33:
      v31 = v30[1];
      if ( !v31 || v24 != **(_QWORD **)(v31 + 32) )
        goto LABEL_35;
      if ( v24 == v29 )
      {
LABEL_47:
        v37 = v28[1];
      }
      else
      {
        v48 = 1;
        while ( v29 != -8 )
        {
          v51 = v48 + 1;
          v27 = v26 & (v48 + v27);
          v28 = (__int64 *)(v25 + 16LL * v27);
          v29 = *v28;
          if ( v24 == *v28 )
            goto LABEL_47;
          v48 = v51;
        }
        v37 = 0;
      }
      if ( (_DWORD)v21 != 2 )
        goto LABEL_35;
      v57 = (_QWORD *)(v14 - 48);
      if ( v22 )
        v57 = *(_QWORD **)(v14 - 8);
      v38 = *v57;
      if ( *(_BYTE *)(*v57 + 16LL) <= 0x17u )
        goto LABEL_54;
      v39 = *(_QWORD *)(v38 + 40);
      v61 = v26 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v40 = (__int64 *)(v25 + 16LL * v61);
      v41 = *v40;
      v42 = v40;
      if ( v39 != *v40 )
      {
        v56 = *v40;
        v59 = v26 & (((unsigned int)*(_QWORD *)(v38 + 40) >> 9) ^ ((unsigned int)v39 >> 4));
        v49 = 1;
        while ( v56 != -8 )
        {
          v54 = v49 + 1;
          v55 = v26 & (v59 + v49);
          v59 = v55;
          v42 = (__int64 *)(v25 + 16 * v55);
          v56 = *v42;
          if ( v39 == *v42 )
          {
            v40 = (__int64 *)(v25 + 16LL * v61);
            goto LABEL_53;
          }
          v49 = v54;
        }
        if ( v37 )
          goto LABEL_54;
        goto LABEL_73;
      }
LABEL_53:
      if ( v37 != v42[1] )
      {
LABEL_54:
        v38 = v57[3];
        if ( !v38 )
          BUG();
        if ( *(_BYTE *)(v38 + 16) <= 0x17u )
          goto LABEL_35;
        v39 = *(_QWORD *)(v38 + 40);
        v61 = v26 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v40 = (__int64 *)(v25 + 16LL * v61);
        v41 = *v40;
      }
      if ( v39 == v41 )
      {
LABEL_58:
        v43 = v40[1];
        goto LABEL_59;
      }
LABEL_73:
      v50 = 1;
      while ( v41 != -8 )
      {
        v52 = v50 + 1;
        v53 = v26 & (v61 + v50);
        v61 = v53;
        v40 = (__int64 *)(v25 + 16 * v53);
        v41 = *v40;
        if ( v39 == *v40 )
          goto LABEL_58;
        v50 = v52;
      }
      v43 = 0;
LABEL_59:
      if ( v37 != v43 || *(_BYTE *)(v38 + 16) != 54 )
        goto LABEL_35;
      if ( sub_13FC1A0(v37, *(_QWORD *)(v38 - 24)) )
      {
        v22 = *(_BYTE *)(v14 + 23) & 0x40;
        v21 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
LABEL_35:
        v32 = 3 * v21;
        if ( v22 )
        {
          v33 = *(__int64 **)(v14 - 8);
          v14 = (__int64)&v33[v32];
        }
        else
        {
          v33 = (__int64 *)(v14 - v32 * 8);
        }
        v9 = (unsigned int)v66;
        if ( (__int64 *)v14 != v33 )
        {
          v60 = a5;
          v34 = a3;
          v35 = (__int64 *)v14;
          do
          {
            v36 = *v33;
            if ( HIDWORD(v66) <= (unsigned int)v9 )
            {
              sub_16CD150(&v65, v67, 0, 8);
              v9 = (unsigned int)v66;
            }
            v33 += 3;
            v65[v9] = v36;
            v9 = (unsigned int)(v66 + 1);
            LODWORD(v66) = v66 + 1;
          }
          while ( v35 != v33 );
          a3 = v34;
          a5 = v60;
        }
        goto LABEL_10;
      }
LABEL_9:
      LODWORD(v9) = v66;
LABEL_10:
      v8 = v65;
      if ( !(_DWORD)v9 )
        goto LABEL_25;
    }
    v18 = *(_QWORD *)(v14 - 48);
    v19 = (unsigned int)v66;
    if ( (unsigned int)v66 >= HIDWORD(v66) )
    {
      sub_16CD150(&v65, v67, 0, 8);
      v19 = (unsigned int)v66;
    }
    v65[v19] = v18;
    v9 = (unsigned int)(v66 + 1);
    LODWORD(v66) = v9;
    v20 = *(_QWORD *)(v14 - 24);
    if ( HIDWORD(v66) <= (unsigned int)v9 )
    {
      sub_16CD150(&v65, v67, 0, 8);
      v9 = (unsigned int)v66;
    }
    v65[v9] = v20;
    v8 = v65;
    LODWORD(v9) = v66 + 1;
    LODWORD(v66) = v9;
  }
  while ( (_DWORD)v9 );
LABEL_25:
  if ( v8 != v67 )
    _libc_free((unsigned __int64)v8);
  if ( v70 != v69 )
    _libc_free((unsigned __int64)v70);
}

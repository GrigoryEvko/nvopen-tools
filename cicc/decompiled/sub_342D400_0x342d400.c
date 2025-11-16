// Function: sub_342D400
// Address: 0x342d400
//
void __fastcall sub_342D400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r15
  unsigned int *v13; // rbx
  unsigned int *v14; // r12
  __int64 v15; // rsi
  __int64 *v16; // rax
  _QWORD *v17; // rax
  int v18; // ebx
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rbx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  unsigned int v29; // ecx
  unsigned __int64 v30; // rbx
  bool v31; // cc
  unsigned int v32; // ecx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rcx
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  const void **v39; // r12
  unsigned int v40; // esi
  unsigned int v41; // esi
  unsigned int v42; // esi
  unsigned __int64 v43; // r12
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // r12
  unsigned __int64 v47; // [rsp+0h] [rbp-560h]
  unsigned __int64 v48; // [rsp+0h] [rbp-560h]
  unsigned __int64 v49; // [rsp+8h] [rbp-558h]
  unsigned __int64 v50; // [rsp+8h] [rbp-558h]
  unsigned __int64 v51; // [rsp+10h] [rbp-550h]
  unsigned __int64 v52; // [rsp+10h] [rbp-550h]
  unsigned int v53; // [rsp+18h] [rbp-548h]
  unsigned __int64 v54; // [rsp+18h] [rbp-548h]
  unsigned int v55; // [rsp+18h] [rbp-548h]
  unsigned int v56; // [rsp+18h] [rbp-548h]
  int v57; // [rsp+28h] [rbp-538h]
  __int64 v58; // [rsp+28h] [rbp-538h]
  __int16 v59; // [rsp+30h] [rbp-530h] BYREF
  __int64 v60; // [rsp+38h] [rbp-528h]
  unsigned __int64 v61; // [rsp+40h] [rbp-520h] BYREF
  unsigned int v62; // [rsp+48h] [rbp-518h]
  unsigned __int64 v63; // [rsp+50h] [rbp-510h] BYREF
  unsigned int v64; // [rsp+58h] [rbp-508h]
  unsigned __int64 v65; // [rsp+60h] [rbp-500h] BYREF
  unsigned int v66; // [rsp+68h] [rbp-4F8h]
  unsigned __int64 v67; // [rsp+70h] [rbp-4F0h]
  unsigned int v68; // [rsp+78h] [rbp-4E8h]
  __int64 v69; // [rsp+80h] [rbp-4E0h] BYREF
  __int64 *v70; // [rsp+88h] [rbp-4D8h]
  __int64 v71; // [rsp+90h] [rbp-4D0h]
  int v72; // [rsp+98h] [rbp-4C8h]
  char v73; // [rsp+9Ch] [rbp-4C4h]
  __int64 v74; // [rsp+A0h] [rbp-4C0h] BYREF
  __int64 *v75; // [rsp+120h] [rbp-440h] BYREF
  __int64 v76; // [rsp+128h] [rbp-438h]
  _QWORD v77[134]; // [rsp+130h] [rbp-430h] BYREF

  v7 = v77;
  v70 = &v74;
  v8 = *(_QWORD *)(a1 + 64);
  v76 = 0x8000000001LL;
  v71 = 0x100000010LL;
  v9 = *(_QWORD *)(v8 + 384);
  v75 = v77;
  v72 = 0;
  v73 = 1;
  v69 = 1;
  v62 = 1;
  v61 = 0;
  v64 = 1;
  v63 = 0;
  v77[0] = v9;
  v74 = v9;
  v10 = 1;
  while ( 1 )
  {
    v11 = v10;
    v12 = v7[v10 - 1];
    LODWORD(v76) = v10 - 1;
    v13 = *(unsigned int **)(v12 + 40);
    v14 = &v13[10 * *(unsigned int *)(v12 + 64)];
    if ( v14 != v13 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)v13;
        if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * v13[2]) != 1 )
          goto LABEL_9;
        if ( v73 )
        {
          v16 = v70;
          v11 = HIDWORD(v71);
          v7 = &v70[HIDWORD(v71)];
          if ( v70 != v7 )
          {
            while ( v15 != *v16 )
            {
              if ( v7 == ++v16 )
                goto LABEL_55;
            }
            goto LABEL_9;
          }
LABEL_55:
          if ( HIDWORD(v71) < (unsigned int)v71 )
          {
            ++HIDWORD(v71);
            *v7 = v15;
            ++v69;
            goto LABEL_51;
          }
        }
        sub_C8CC70((__int64)&v69, v15, (__int64)v7, v11, a5, a6);
        if ( (_BYTE)v7 )
        {
LABEL_51:
          v33 = (unsigned int)v76;
          v11 = HIDWORD(v76);
          a6 = *(_QWORD *)v13;
          v34 = (unsigned int)v76 + 1LL;
          if ( v34 > HIDWORD(v76) )
          {
            v58 = *(_QWORD *)v13;
            sub_C8D5F0((__int64)&v75, v77, v34, 8u, a5, a6);
            v33 = (unsigned int)v76;
            a6 = v58;
          }
          v7 = v75;
          v13 += 10;
          v75[v33] = a6;
          LODWORD(v76) = v76 + 1;
          if ( v14 == v13 )
            break;
        }
        else
        {
LABEL_9:
          v13 += 10;
          if ( v14 == v13 )
            break;
        }
      }
    }
    if ( *(_DWORD *)(v12 + 24) != 49 )
      goto LABEL_31;
    v17 = *(_QWORD **)(v12 + 40);
    v18 = *(_DWORD *)(v17[5] + 96LL);
    if ( v18 >= 0 )
      goto LABEL_31;
    v19 = v17[10];
    v20 = v17[11];
    v21 = *(_QWORD *)(v19 + 48) + 16LL * *((unsigned int *)v17 + 22);
    v22 = *(_WORD *)v21;
    v23 = *(_QWORD *)(v21 + 8);
    v59 = v22;
    v60 = v23;
    if ( v22 )
    {
      if ( (unsigned __int16)(v22 - 2) > 7u
        && (unsigned __int16)(v22 - 17) > 0x6Cu
        && (unsigned __int16)(v22 - 176) > 0x1Fu )
      {
        goto LABEL_31;
      }
    }
    else if ( !sub_3007070((__int64)&v59) )
    {
      goto LABEL_31;
    }
    v57 = sub_33D4D80(*(_QWORD *)(a1 + 64), v19, v20, 0);
    sub_33DD090((__int64)&v65, *(_QWORD *)(a1 + 64), v19, v20, 0);
    if ( v62 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
    v61 = v65;
    v24 = v66;
    v66 = 0;
    v62 = v24;
    if ( v64 > 0x40 && v63 )
    {
      j_j___libc_free_0_0(v63);
      v63 = v67;
      v64 = v68;
      if ( v66 > 0x40 && v65 )
      {
        j_j___libc_free_0_0(v65);
        v25 = *(_QWORD *)(a1 + 24);
        if ( v57 != 1 )
          goto LABEL_24;
        goto LABEL_35;
      }
    }
    else
    {
      v63 = v67;
      v64 = v68;
    }
    v25 = *(_QWORD *)(a1 + 24);
    if ( v57 != 1 )
      goto LABEL_24;
LABEL_35:
    if ( v62 > 0x40 )
    {
      v53 = v62;
      if ( v53 != (unsigned int)sub_C444A0((__int64)&v61) )
        goto LABEL_24;
      v32 = v64;
      if ( v64 > 0x40 )
        break;
      goto LABEL_62;
    }
    if ( v61 )
      goto LABEL_24;
    v32 = v64;
    if ( v64 > 0x40 )
      break;
LABEL_62:
    if ( v63 )
      goto LABEL_24;
LABEL_31:
    v10 = v76;
    if ( !(_DWORD)v76 )
      goto LABEL_40;
LABEL_32:
    v7 = v75;
  }
  if ( v32 != (unsigned int)sub_C444A0((__int64)&v63) )
  {
LABEL_24:
    v26 = v18 & 0x7FFFFFFF;
    v27 = *(unsigned int *)(v25 + 1096);
    v28 = *(_QWORD *)(v25 + 1088);
    v29 = v26 + 1;
    if ( (int)v26 + 1 > (unsigned int)v27 )
    {
      a5 = v29;
      if ( v29 != v27 )
      {
        v35 = v28 + 40 * v27;
        if ( v29 >= v27 )
        {
          a6 = v25 + 1088;
          v36 = v25 + 1104;
          v37 = a5 - v27;
          v54 = a5 - v27;
          if ( a5 > *(unsigned int *)(v25 + 1100) )
          {
            if ( v28 > v36 || v35 <= v36 )
            {
              sub_342D2F0(v25 + 1088, a5, v37, v36, a5, a6);
              v28 = *(_QWORD *)(v25 + 1088);
              v27 = *(unsigned int *)(v25 + 1096);
              v36 = v25 + 1104;
            }
            else
            {
              v46 = v36 - v28;
              sub_342D2F0(v25 + 1088, a5, v37, v36 - v28, a5, a6);
              v28 = *(_QWORD *)(v25 + 1088);
              v27 = *(unsigned int *)(v25 + 1096);
              v36 = v28 + v46;
            }
          }
          a5 = v54;
          v38 = v28 + 40 * v27;
          v39 = (const void **)(v36 + 24);
          while ( 1 )
          {
LABEL_72:
            if ( !v38 )
              goto LABEL_71;
            *(_DWORD *)v38 = *(_DWORD *)v36;
            v41 = *(_DWORD *)(v36 + 16);
            *(_DWORD *)(v38 + 16) = v41;
            if ( v41 > 0x40 )
              break;
            *(_QWORD *)(v38 + 8) = *(_QWORD *)(v36 + 8);
            v40 = *(_DWORD *)(v36 + 32);
            *(_DWORD *)(v38 + 32) = v40;
            if ( v40 <= 0x40 )
              goto LABEL_70;
LABEL_75:
            v48 = a5;
            v50 = v36;
            v52 = v38;
            sub_C43780(v38 + 24, v39);
            v36 = v50;
            v38 = v52 + 40;
            a5 = v48 - 1;
            if ( v48 == 1 )
            {
LABEL_76:
              v28 = *(_QWORD *)(v25 + 1088);
              *(_DWORD *)(v25 + 1096) += v54;
              goto LABEL_25;
            }
          }
          v47 = a5;
          v49 = v36;
          v51 = v38;
          sub_C43780(v38 + 8, (const void **)(v36 + 8));
          v36 = v49;
          v38 = v51;
          a5 = v47;
          v42 = *(_DWORD *)(v49 + 32);
          *(_DWORD *)(v51 + 32) = v42;
          if ( v42 > 0x40 )
            goto LABEL_75;
LABEL_70:
          *(_QWORD *)(v38 + 24) = *(_QWORD *)(v36 + 24);
LABEL_71:
          v38 += 40LL;
          if ( !--a5 )
            goto LABEL_76;
          goto LABEL_72;
        }
        v43 = v28 + 40LL * v29;
        if ( v35 != v43 )
        {
          do
          {
            v35 -= 40LL;
            if ( *(_DWORD *)(v35 + 32) > 0x40u )
            {
              v44 = *(_QWORD *)(v35 + 24);
              if ( v44 )
              {
                v55 = v29;
                j_j___libc_free_0_0(v44);
                v29 = v55;
              }
            }
            if ( *(_DWORD *)(v35 + 16) > 0x40u )
            {
              v45 = *(_QWORD *)(v35 + 8);
              if ( v45 )
              {
                v56 = v29;
                j_j___libc_free_0_0(v45);
                v29 = v56;
              }
            }
          }
          while ( v43 != v35 );
          v28 = *(_QWORD *)(v25 + 1088);
        }
        *(_DWORD *)(v25 + 1096) = v29;
      }
    }
LABEL_25:
    v30 = v28 + 40 * v26;
    v31 = *(_DWORD *)(v30 + 32) <= 0x40u;
    *(_DWORD *)v30 = *(_DWORD *)v30 & 0x80000000 | v57 & 0x7FFFFFFF;
    if ( v31 && v64 <= 0x40 )
    {
      *(_QWORD *)(v30 + 24) = v63;
      *(_DWORD *)(v30 + 32) = v64;
    }
    else
    {
      sub_C43990(v30 + 24, (__int64)&v63);
    }
    if ( *(_DWORD *)(v30 + 16) <= 0x40u && v62 <= 0x40 )
    {
      *(_QWORD *)(v30 + 8) = v61;
      *(_DWORD *)(v30 + 16) = v62;
    }
    else
    {
      sub_C43990(v30 + 8, (__int64)&v61);
    }
    goto LABEL_31;
  }
  v10 = v76;
  if ( (_DWORD)v76 )
    goto LABEL_32;
LABEL_40:
  if ( v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
  if ( !v73 )
    _libc_free((unsigned __int64)v70);
}

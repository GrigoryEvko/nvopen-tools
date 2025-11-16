// Function: sub_310B7A0
// Address: 0x310b7a0
//
void __fastcall sub_310B7A0(__int64 **a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 *v5; // rbx
  bool v6; // r13
  bool v7; // al
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r13
  bool v13; // r8
  unsigned __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // rax
  _BYTE *v19; // rdi
  __int64 *v20; // r13
  __int64 v21; // r12
  int v22; // r9d
  _QWORD *v23; // rdi
  unsigned int v24; // esi
  _QWORD *v25; // rax
  __int64 v26; // r10
  __int64 *v27; // rax
  __int64 *v28; // r13
  __int64 *v29; // rax
  __int64 v30; // r15
  __int64 *v31; // rax
  __int64 *v32; // r8
  unsigned int v33; // esi
  __int64 v34; // r15
  unsigned int v35; // ecx
  _QWORD *v36; // rdx
  __int64 v37; // rdi
  __int64 **v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rbx
  bool v41; // zf
  __int64 *v42; // rax
  _QWORD *v43; // r13
  int v44; // ebx
  _QWORD *v45; // rax
  int v46; // edx
  int v47; // r10d
  unsigned int v48; // ecx
  _QWORD *v49; // rsi
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // rdi
  int v53; // r10d
  __int64 *v55; // [rsp+18h] [rbp-F8h]
  bool v56; // [rsp+20h] [rbp-F0h]
  int v57; // [rsp+20h] [rbp-F0h]
  __int64 *v58; // [rsp+20h] [rbp-F0h]
  __int64 *v59; // [rsp+28h] [rbp-E8h]
  __int64 *v60; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD *v61; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v62; // [rsp+48h] [rbp-C8h]
  _BYTE v63[16]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+68h] [rbp-A8h]
  __int64 v66; // [rsp+70h] [rbp-A0h]
  unsigned int v67; // [rsp+78h] [rbp-98h]
  __int64 *v68; // [rsp+80h] [rbp-90h] BYREF
  __int64 v69; // [rsp+88h] [rbp-88h]
  __int64 v70; // [rsp+90h] [rbp-80h]
  __int64 v71; // [rsp+98h] [rbp-78h] BYREF
  unsigned int v72; // [rsp+A0h] [rbp-70h]
  _QWORD v73[7]; // [rsp+D8h] [rbp-38h] BYREF

  v3 = (__int64)a1[1];
  v61 = v63;
  v62 = 0x200000000LL;
  v4 = sub_D95540(v3);
  v5 = *(__int64 **)(a2 + 32);
  v59 = &v5[*(_QWORD *)(a2 + 40)];
  if ( v59 != v5 )
  {
    v6 = 0;
    do
    {
      while ( 1 )
      {
        v15 = *v5;
        if ( v4 != sub_D95540(*v5) )
          goto LABEL_15;
        if ( !v6 )
        {
          sub_310BF50(*a1, v15, a1[1], &v64, &v68);
          v7 = sub_D968A0((__int64)v68);
          if ( v7 )
            break;
        }
        v16 = (unsigned int)v62;
        v17 = (unsigned int)v62 + 1LL;
        if ( v17 > HIDWORD(v62) )
        {
          sub_C8D5F0((__int64)&v61, v63, v17, 8u, v8, v9);
          v16 = (unsigned int)v62;
        }
        ++v5;
        v61[v16] = v15;
        LODWORD(v62) = v62 + 1;
        if ( v59 == v5 )
          goto LABEL_13;
      }
      v56 = v7;
      if ( v4 != sub_D95540(v64) )
        goto LABEL_15;
      v11 = (unsigned int)v62;
      v12 = v64;
      v13 = v56;
      v14 = (unsigned int)v62 + 1LL;
      if ( v14 > HIDWORD(v62) )
      {
        sub_C8D5F0((__int64)&v61, v63, v14, 8u, v56, v10);
        v11 = (unsigned int)v62;
        v13 = v56;
      }
      ++v5;
      v61[v11] = v12;
      v6 = v13;
      LODWORD(v62) = v62 + 1;
    }
    while ( v59 != v5 );
LABEL_13:
    if ( v6 )
    {
      v41 = (_DWORD)v62 == 1;
      a1[3] = a1[4];
      if ( v41 )
      {
        v19 = v61;
        a1[2] = (__int64 *)*v61;
      }
      else
      {
        v42 = sub_DC8BD0(*a1, (__int64)&v61, 0, 0);
        v19 = v61;
        a1[2] = v42;
      }
      goto LABEL_16;
    }
  }
  v18 = a1[1];
  if ( *((_WORD *)v18 + 12) != 15 )
  {
LABEL_15:
    sub_310A840(a1, a2);
    v19 = v61;
    goto LABEL_16;
  }
  v65 = 0;
  v20 = a1[4];
  v66 = 0;
  v67 = 0;
  v21 = *(v18 - 1);
  v64 = 1;
  sub_310A8E0((__int64)&v64, 0);
  if ( !v67 )
  {
    LODWORD(v66) = v66 + 1;
    BUG();
  }
  v22 = 1;
  v23 = 0;
  v24 = (v67 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v25 = (_QWORD *)(v65 + 16LL * v24);
  v26 = *v25;
  if ( v21 != *v25 )
  {
    while ( v26 != -4096 )
    {
      if ( v26 == -8192 && !v23 )
        v23 = v25;
      v24 = (v67 - 1) & (v22 + v24);
      v25 = (_QWORD *)(v65 + 16LL * v24);
      v26 = *v25;
      if ( v21 == *v25 )
        goto LABEL_21;
      ++v22;
    }
    if ( v23 )
      v25 = v23;
  }
LABEL_21:
  LODWORD(v66) = v66 + 1;
  if ( *v25 != -4096 )
    --HIDWORD(v66);
  *v25 = v21;
  v25[1] = v20;
  v27 = *a1;
  v28 = &v71;
  v69 = 0;
  v70 = 1;
  v68 = v27;
  v29 = &v71;
  do
  {
    *v29 = -4096;
    v29 += 2;
  }
  while ( v29 != v73 );
  v73[0] = &v64;
  v30 = sub_310AAC0((__int64)&v68, a2);
  if ( (v70 & 1) == 0 )
    sub_C7D6A0(v71, 16LL * v72, 8);
  a1[3] = (__int64 *)v30;
  if ( sub_D968A0(v30) )
  {
    v31 = a1[1];
    v32 = a1[5];
    if ( !v31 )
      BUG();
    v33 = v67;
    v34 = *(v31 - 1);
    if ( v67 )
    {
      v35 = (v67 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v36 = (_QWORD *)(v65 + 16LL * v35);
      v37 = *v36;
      if ( v34 == *v36 )
      {
LABEL_31:
        v38 = (__int64 **)(v36 + 1);
LABEL_32:
        *v38 = v32;
        v39 = *a1;
        v69 = 0;
        v68 = v39;
        v70 = 1;
        do
        {
          *v28 = -4096;
          v28 += 2;
        }
        while ( v28 != v73 );
        v73[0] = &v64;
        v40 = sub_310AAC0((__int64)&v68, a2);
        if ( (v70 & 1) == 0 )
          sub_C7D6A0(v71, 16LL * v72, 8);
        a1[2] = (__int64 *)v40;
        goto LABEL_37;
      }
      v57 = 1;
      v45 = 0;
      while ( v37 != -4096 )
      {
        if ( v37 == -8192 && !v45 )
          v45 = v36;
        v35 = (v67 - 1) & (v57 + v35);
        v36 = (_QWORD *)(v65 + 16LL * v35);
        v37 = *v36;
        if ( v34 == *v36 )
          goto LABEL_31;
        ++v57;
      }
      v33 = v67;
      if ( !v45 )
        v45 = v36;
      ++v64;
      v46 = v66 + 1;
      if ( 4 * ((int)v66 + 1) < 3 * v67 )
      {
        if ( v67 - HIDWORD(v66) - v46 > v67 >> 3 )
        {
LABEL_51:
          LODWORD(v66) = v46;
          if ( *v45 != -4096 )
            --HIDWORD(v66);
          *v45 = v34;
          v38 = (__int64 **)(v45 + 1);
          *v38 = 0;
          goto LABEL_32;
        }
        v55 = v32;
        sub_310A8E0((__int64)&v64, v67);
        if ( v67 )
        {
          v47 = 1;
          v32 = v55;
          v48 = (v67 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
          v46 = v66 + 1;
          v49 = 0;
          v45 = (_QWORD *)(v65 + 16LL * v48);
          v50 = *v45;
          if ( v34 == *v45 )
            goto LABEL_51;
          while ( v50 != -4096 )
          {
            if ( !v49 && v50 == -8192 )
              v49 = v45;
            v48 = (v67 - 1) & (v47 + v48);
            v45 = (_QWORD *)(v65 + 16LL * v48);
            v50 = *v45;
            if ( v34 == *v45 )
              goto LABEL_51;
            ++v47;
          }
LABEL_57:
          if ( v49 )
            v45 = v49;
          goto LABEL_51;
        }
        goto LABEL_90;
      }
    }
    else
    {
      ++v64;
    }
    v58 = v32;
    sub_310A8E0((__int64)&v64, 2 * v33);
    if ( v67 )
    {
      v32 = v58;
      v51 = (v67 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v46 = v66 + 1;
      v45 = (_QWORD *)(v65 + 16LL * v51);
      v52 = *v45;
      if ( v34 == *v45 )
        goto LABEL_51;
      v53 = 1;
      v49 = 0;
      while ( v52 != -4096 )
      {
        if ( !v49 && v52 == -8192 )
          v49 = v45;
        v51 = (v67 - 1) & (v53 + v51);
        v45 = (_QWORD *)(v65 + 16LL * v51);
        v52 = *v45;
        if ( v34 == *v45 )
          goto LABEL_51;
        ++v53;
      }
      goto LABEL_57;
    }
LABEL_90:
    LODWORD(v66) = v66 + 1;
    BUG();
  }
  v43 = sub_DCC810(*a1, a2, (__int64)a1[3], 0, 0);
  v44 = sub_310A400((__int64)v43);
  if ( v44 <= (int)sub_310A400(a2) && (sub_310BF50(*a1, v43, a1[1], &v60, &v68), a1[4] == v68) )
    a1[2] = v60;
  else
    sub_310A840(a1, a2);
LABEL_37:
  sub_C7D6A0(v65, 16LL * v67, 8);
  v19 = v61;
LABEL_16:
  if ( v19 != v63 )
    _libc_free((unsigned __int64)v19);
}

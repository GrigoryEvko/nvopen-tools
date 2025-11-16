// Function: sub_3521900
// Address: 0x3521900
//
void __fastcall sub_3521900(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 *v3; // r14
  __int64 (*v4)(); // r13
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  _QWORD **v16; // r12
  _QWORD **i; // r14
  _QWORD *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rax
  __int64 v26; // r10
  __int64 v27; // r12
  _QWORD *v28; // rcx
  __int64 v29; // r12
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // rdx
  _QWORD *v33; // rsi
  _QWORD *v34; // rax
  __int64 **v35; // r12
  __int64 v36; // rax
  __int64 v37; // r11
  unsigned __int64 *v38; // r14
  __int64 ***v39; // r15
  __int64 (*v40)(); // r10
  __int64 v41; // r13
  __int64 **v42; // rbx
  unsigned __int64 *v43; // rdx
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // r11
  __int64 (*v48)(); // rax
  __int64 *v49; // rax
  char v50; // al
  __int64 v51; // rdi
  __int64 (*v52)(); // r10
  _QWORD *v53; // rdi
  unsigned __int64 v54; // rdi
  __int64 v55; // [rsp+8h] [rbp-198h]
  __int64 *v56; // [rsp+10h] [rbp-190h]
  __int64 v57; // [rsp+18h] [rbp-188h]
  __int64 v58; // [rsp+30h] [rbp-170h]
  unsigned __int64 v59; // [rsp+30h] [rbp-170h]
  __int64 (*v60)(); // [rsp+38h] [rbp-168h]
  __int64 v61; // [rsp+40h] [rbp-160h] BYREF
  __int64 v62; // [rsp+48h] [rbp-158h] BYREF
  _QWORD *v63; // [rsp+50h] [rbp-150h] BYREF
  __int64 v64; // [rsp+58h] [rbp-148h]
  _QWORD v65[4]; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v66; // [rsp+80h] [rbp-120h] BYREF
  char *v67; // [rsp+88h] [rbp-118h]
  __int64 v68; // [rsp+90h] [rbp-110h]
  int v69; // [rsp+98h] [rbp-108h]
  char v70; // [rsp+9Ch] [rbp-104h]
  char v71; // [rsp+A0h] [rbp-100h] BYREF
  _BYTE *v72; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v73; // [rsp+C8h] [rbp-D8h]
  _BYTE v74[208]; // [rsp+D0h] [rbp-D0h] BYREF

  v1 = a1;
  v72 = v74;
  v73 = 0x400000000LL;
  v2 = *(_QWORD *)(a1 + 520);
  v57 = a1 + 888;
  v3 = *(__int64 **)(v2 + 328);
  v56 = (__int64 *)(v2 + 320);
  if ( v3 == (__int64 *)(v2 + 320) )
  {
    v4 = sub_2DB1AE0;
  }
  else
  {
    v4 = sub_2DB1AE0;
    v55 = a1 + 792;
    do
    {
      v5 = *(_QWORD *)(v1 + 792);
      *(_QWORD *)(v1 + 872) += 64LL;
      v6 = ((v5 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 64;
      if ( *(_QWORD *)(v1 + 800) >= v6 && v5 )
      {
        *(_QWORD *)(v1 + 792) = v6;
        v7 = (v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v7 = sub_9D1E70(v55, 64, 64, 3);
      }
      v66 = v3;
      *(_QWORD *)v7 = v7 + 16;
      v8 = (__int64)v66;
      *(_DWORD *)(v7 + 56) = 0;
      *(_QWORD *)(v7 + 16) = v8;
      *(_QWORD *)(v7 + 8) = 0x400000001LL;
      *(_QWORD *)(v7 + 48) = v57;
      *sub_3515040(v57, (__int64 *)&v66) = v7;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v1 + 560);
        v62 = 0;
        v63 = 0;
        LODWORD(v73) = 0;
        v14 = *(__int64 (**)())(*(_QWORD *)v13 + 344LL);
        if ( v14 != sub_2DB1AE0
          && !((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64 *, _QWORD **, _BYTE **, _QWORD))v14)(
                v13,
                v3,
                &v62,
                &v63,
                &v72,
                0) )
        {
          break;
        }
        if ( !sub_2E32580(v3) )
          break;
        v3 = (__int64 *)v3[1];
        v66 = v3;
        v11 = *(unsigned int *)(v7 + 8);
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 12) )
        {
          sub_C8D5F0(v7, (const void *)(v7 + 16), v11 + 1, 8u, v9, v10);
          v11 = *(unsigned int *)(v7 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v7 + 8 * v11) = v3;
        v12 = *(_QWORD *)(v7 + 48);
        ++*(_DWORD *)(v7 + 8);
        *sub_3515040(v12, (__int64 *)&v66) = v7;
      }
      v3 = (__int64 *)v3[1];
    }
    while ( v3 != v56 );
  }
  v15 = *(_QWORD *)(v1 + 544);
  *(_QWORD *)(v1 + 552) = 0;
  v16 = *(_QWORD ***)(v15 + 40);
  for ( i = *(_QWORD ***)(v15 + 32); v16 != i; ++i )
  {
    v18 = *i;
    sub_351EBB0((_QWORD *)v1, v18);
  }
  v66 = 0;
  v67 = &v71;
  v19 = *(_QWORD *)(v1 + 520);
  v68 = 4;
  v70 = 1;
  v20 = v19 + 320;
  v69 = 0;
  v21 = *(_QWORD *)(v19 + 328);
  if ( v19 + 320 != v21 )
  {
    do
    {
      sub_35157A0(v1, v21, (__int64)&v66, 0);
      v21 = *(_QWORD *)(v21 + 8);
    }
    while ( v21 != v20 );
    v21 = *(_QWORD *)(*(_QWORD *)(v1 + 520) + 328LL);
  }
  v63 = (_QWORD *)v21;
  v22 = sub_3515040(v57, (__int64 *)&v63);
  v58 = *v22;
  sub_351D700(v1, *(_QWORD *)(*(_QWORD *)(v1 + 520) + 328LL), *v22, 0);
  v25 = *(_QWORD **)(v1 + 520);
  v26 = v58;
  v27 = v25[13] - v25[12];
  v64 = 0x400000000LL;
  v28 = v65;
  v29 = v27 >> 3;
  v63 = v65;
  if ( (_DWORD)v29 )
  {
    v30 = v65;
    if ( (unsigned int)v29 > 4uLL )
    {
      sub_C8D5F0((__int64)&v63, v65, (unsigned int)v29, 8u, v23, v24);
      v28 = v63;
      v26 = v58;
      v30 = &v63[(unsigned int)v64];
      v31 = &v63[(unsigned int)v29];
      if ( v30 != v31 )
        goto LABEL_22;
    }
    else
    {
      v31 = &v65[(unsigned int)v29];
      if ( v65 != v31 )
      {
        do
        {
LABEL_22:
          if ( v30 )
            *v30 = 0;
          ++v30;
        }
        while ( v31 != v30 );
        v28 = v63;
      }
    }
    LODWORD(v64) = v29;
    v25 = *(_QWORD **)(v1 + 520);
  }
  v32 = (_QWORD *)v25[41];
  v33 = v25 + 40;
  if ( v25 + 40 != v32 )
  {
    while ( 1 )
    {
      v34 = (_QWORD *)v32[1];
      if ( v33 == v34 )
        break;
      v28[*((int *)v32 + 6)] = v34;
      v28 = v63;
      v32 = v34;
    }
    v25 = *(_QWORD **)(v1 + 520);
  }
  v28[*(int *)((v25[40] & 0xFFFFFFFFFFFFFFF8LL) + 24)] = 0;
  v35 = *(__int64 ***)v26;
  v36 = *(_QWORD *)(v1 + 520);
  v37 = *(_QWORD *)v26 + 8LL * *(unsigned int *)(v26 + 8);
  v38 = *(unsigned __int64 **)(v36 + 328);
  if ( v37 != *(_QWORD *)v26 )
  {
    v39 = (__int64 ***)v26;
    v40 = sub_2DB1AE0;
    v41 = v1;
    v42 = (__int64 **)v37;
    do
    {
      while ( 1 )
      {
        v49 = *v35;
        if ( *v35 == (__int64 *)v38 )
        {
          v38 = (unsigned __int64 *)v38[1];
        }
        else
        {
          v43 = (unsigned __int64 *)v49[1];
          if ( v38 != v43 && v43 != (unsigned __int64 *)v49 )
          {
            v44 = *v43 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v49 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v43;
            *v43 = *v43 & 7 | *v49 & 0xFFFFFFFFFFFFFFF8LL;
            v45 = *v38;
            *(_QWORD *)(v44 + 8) = v38;
            v45 &= 0xFFFFFFFFFFFFFFF8LL;
            *v49 = v45 | *v49 & 7;
            *(_QWORD *)(v45 + 8) = v49;
            *v38 = v44 | *v38 & 7;
          }
        }
        if ( **v39 != v49 )
        {
          v46 = *(_QWORD *)(v41 + 560);
          v47 = *v49;
          v61 = 0;
          v62 = 0;
          LODWORD(v73) = 0;
          v48 = *(__int64 (**)())(*(_QWORD *)v46 + 344LL);
          if ( v48 != v40 )
          {
            v60 = v40;
            v59 = v47 & 0xFFFFFFFFFFFFFFF8LL;
            v50 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v48)(
                    v46,
                    v47 & 0xFFFFFFFFFFFFFFF8LL,
                    &v61,
                    &v62,
                    &v72,
                    0);
            v40 = v60;
            if ( !v50 )
              break;
          }
        }
        if ( v42 == ++v35 )
          goto LABEL_45;
      }
      ++v35;
      sub_2E32A60(v59, v63[*(int *)(v59 + 24)]);
      v40 = v60;
    }
    while ( v42 != v35 );
LABEL_45:
    v1 = v41;
    v4 = v40;
    v36 = *(_QWORD *)(v1 + 520);
  }
  v51 = *(_QWORD *)(v1 + 560);
  v61 = 0;
  v62 = 0;
  LODWORD(v73) = 0;
  v52 = *(__int64 (**)())(*(_QWORD *)v51 + 344LL);
  if ( v52 != v4
    && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v52)(
          v51,
          *(_QWORD *)(v36 + 320) & 0xFFFFFFFFFFFFFFF8LL,
          &v61,
          &v62,
          &v72,
          0) )
  {
    v54 = *(_QWORD *)(*(_QWORD *)(v1 + 520) + 320LL) & 0xFFFFFFFFFFFFFFF8LL;
    sub_2E32A60(v54, v63[*(int *)(v54 + 24)]);
  }
  *(_DWORD *)(v1 + 208) = 0;
  v53 = v63;
  *(_DWORD *)(v1 + 352) = 0;
  if ( v53 != v65 )
    _libc_free((unsigned __int64)v53);
  if ( !v70 )
    _libc_free((unsigned __int64)v67);
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
}

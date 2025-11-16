// Function: sub_1172080
// Address: 0x1172080
//
_QWORD *__fastcall sub_1172080(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // rdi
  __int64 v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 *v9; // r14
  __int64 *v10; // r13
  const void **v11; // r15
  __int64 v13; // rdx
  size_t v14; // rdx
  __int64 v15; // rbx
  unsigned int i; // eax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r13
  const char *v20; // rax
  int v21; // r14d
  int v22; // r14d
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r13
  int v26; // eax
  _QWORD *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rsi
  _QWORD *v31; // r15
  __int64 v32; // r14
  __int64 v33; // rcx
  __int64 *v34; // r14
  __int64 *v35; // rcx
  __int64 v36; // rdi
  __int64 v37; // r13
  __int64 v38; // r8
  __int64 v39; // rdx
  __int64 *v40; // rbx
  __int64 v41; // r12
  __int64 v42; // r14
  int v43; // eax
  int v44; // eax
  unsigned int v45; // r9d
  __int64 v46; // rax
  __int64 v47; // r9
  __int64 v48; // r9
  const char *v49; // rax
  __int64 v50; // r13
  __int64 v51; // r15
  __int64 v52; // rdx
  const void *v53; // r14
  __int64 v54; // rbx
  _QWORD *v55; // rax
  __int64 v56; // [rsp+0h] [rbp-E0h]
  __int64 v57; // [rsp+8h] [rbp-D8h]
  __int64 v58; // [rsp+18h] [rbp-C8h]
  __int64 *v59; // [rsp+18h] [rbp-C8h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+30h] [rbp-B0h]
  __int64 v62; // [rsp+40h] [rbp-A0h]
  __int64 v63; // [rsp+40h] [rbp-A0h]
  __int64 *v64; // [rsp+50h] [rbp-90h]
  __int64 v65; // [rsp+50h] [rbp-90h]
  _QWORD *v66; // [rsp+50h] [rbp-90h]
  __int64 v68; // [rsp+68h] [rbp-78h] BYREF
  _QWORD v69[2]; // [rsp+70h] [rbp-70h] BYREF
  const char *v70; // [rsp+80h] [rbp-60h] BYREF
  __int64 v71; // [rsp+88h] [rbp-58h]
  const char *v72; // [rsp+90h] [rbp-50h]
  __int16 v73; // [rsp+A0h] [rbp-40h]

  v2 = a2;
  v3 = *(__int64 **)(a2 - 8);
  v4 = *v3;
  v5 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
  {
    v6 = (__int64)&v3[v5];
  }
  else
  {
    v3 = (__int64 *)(v2 - v5 * 8);
    v6 = v2;
  }
  v7 = sub_116D080((__int64)v3, v6, 1);
  v9 = v8;
  if ( (__int64 *)v7 == v8 )
  {
LABEL_13:
    v61 = v4;
    v15 = v62;
    v68 = 0x100000000LL;
    v60 = v2 + 24;
    v64 = &v68;
    for ( i = 0; ; i = *(_DWORD *)v64 )
    {
      v17 = (int)i;
      v18 = 32LL * i - 64;
      v19 = v18 + v61;
      v58 = v18;
      v20 = sub_BD5D20(*(_QWORD *)(v18 + v61));
      v21 = *(_DWORD *)(v2 + 4);
      v70 = v20;
      v73 = 773;
      v22 = v21 & 0x7FFFFFF;
      v71 = v23;
      v72 = ".pn";
      v63 = *(_QWORD *)(*(_QWORD *)v19 + 8LL);
      v24 = sub_BD2DA0(80);
      v25 = v24;
      if ( v24 )
      {
        sub_B44260(v24, v63, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v25 + 72) = v22;
        sub_BD6B50((unsigned __int8 *)v25, &v70);
        sub_BD2A10(v25, *(_DWORD *)(v25 + 72), 1);
      }
      v26 = *(_DWORD *)(v2 + 4);
      v27 = *(_QWORD **)(v2 - 8);
      v69[v17] = v25;
      v28 = v26 & 0x7FFFFFF;
      v29 = 4 * v28;
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
      {
        v30 = &v27[v29];
        v31 = v27;
      }
      else
      {
        v31 = (_QWORD *)(v2 - v29 * 8);
        v30 = (_QWORD *)v2;
      }
      v32 = 4LL * *(unsigned int *)(v2 + 72);
      v33 = v32 * 8 + 8 * v28;
      v34 = &v27[v32];
      v35 = (_QWORD *)((char *)v27 + v33);
      if ( v35 != v34 && v31 != v30 )
      {
        v36 = v25;
        v37 = v58;
        v38 = v15;
        v39 = v2;
        v40 = v34;
        do
        {
          v41 = *v40;
          v42 = *(_QWORD *)(*v31 + v37);
          v43 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
          if ( v43 == *(_DWORD *)(v36 + 72) )
          {
            v56 = v38;
            v57 = v39;
            v59 = v35;
            sub_B48D90(v36);
            v38 = v56;
            v39 = v57;
            v35 = v59;
            v43 = *(_DWORD *)(v36 + 4) & 0x7FFFFFF;
          }
          v44 = (v43 + 1) & 0x7FFFFFF;
          v45 = v44 | *(_DWORD *)(v36 + 4) & 0xF8000000;
          v46 = *(_QWORD *)(v36 - 8) + 32LL * (unsigned int)(v44 - 1);
          *(_DWORD *)(v36 + 4) = v45;
          if ( *(_QWORD *)v46 )
          {
            v47 = *(_QWORD *)(v46 + 8);
            **(_QWORD **)(v46 + 16) = v47;
            if ( v47 )
              *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
          }
          *(_QWORD *)v46 = v42;
          if ( v42 )
          {
            v48 = *(_QWORD *)(v42 + 16);
            *(_QWORD *)(v46 + 8) = v48;
            if ( v48 )
              *(_QWORD *)(v48 + 16) = v46 + 8;
            *(_QWORD *)(v46 + 16) = v42 + 16;
            *(_QWORD *)(v42 + 16) = v46;
          }
          v31 += 4;
          ++v40;
          *(_QWORD *)(*(_QWORD *)(v36 - 8)
                    + 32LL * *(unsigned int *)(v36 + 72)
                    + 8LL * ((*(_DWORD *)(v36 + 4) & 0x7FFFFFFu) - 1)) = v41;
        }
        while ( v31 != v30 && v35 != v40 );
        v25 = v36;
        v2 = v39;
        v15 = v38;
      }
      LOWORD(v15) = 0;
      sub_B44220((_QWORD *)v25, v60, v15);
      v70 = (const char *)v25;
      sub_11715E0(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v70);
      v64 = (__int64 *)((char *)v64 + 4);
      if ( v69 == v64 )
        break;
    }
    v49 = sub_BD5D20(v2);
    v50 = v69[1];
    v70 = v49;
    v73 = 261;
    v51 = v69[0];
    v71 = v52;
    v53 = *(const void **)(v61 + 72);
    v54 = *(unsigned int *)(v61 + 80);
    v55 = sub_BD2C40(104, unk_3F148BC);
    if ( v55 )
    {
      v65 = (__int64)v55;
      sub_B44260((__int64)v55, *(_QWORD *)(v51 + 8), 65, 2u, 0, 0);
      *(_QWORD *)(v65 + 80) = 0x400000000LL;
      *(_QWORD *)(v65 + 72) = v65 + 88;
      sub_B4FD20(v65, v51, v50, v53, v54, (__int64)&v70);
      v55 = (_QWORD *)v65;
    }
    v66 = v55;
    sub_116D800(a1, (__int64)v55, v2);
    return v66;
  }
  else
  {
    v10 = (__int64 *)v7;
    while ( 1 )
    {
      v11 = (const void **)*v10;
      if ( *(_BYTE *)*v10 != 94 )
        return 0;
      if ( !(unsigned __int8)sub_BD36B0(*v10) )
        return 0;
      v13 = *((unsigned int *)v11 + 20);
      if ( *(_DWORD *)(v4 + 80) != (_DWORD)v13 )
        return 0;
      v14 = 4 * v13;
      if ( v14 )
      {
        if ( memcmp(v11[9], *(const void **)(v4 + 72), v14) )
          return 0;
      }
      v10 += 4;
      if ( v9 == v10 )
        goto LABEL_13;
    }
  }
}

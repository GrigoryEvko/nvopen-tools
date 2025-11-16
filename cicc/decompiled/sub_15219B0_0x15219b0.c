// Function: sub_15219B0
// Address: 0x15219b0
//
__int64 *__fastcall sub_15219B0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 *v16; // rbx
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  const char *v19; // rax
  unsigned int v20; // ebx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned int v24; // esi
  int *v25; // r13
  int v26; // r9d
  int v27; // r10d
  unsigned __int64 v28; // rcx
  _BYTE *v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // [rsp+0h] [rbp-310h]
  __int64 v37; // [rsp+8h] [rbp-308h]
  __int64 v38; // [rsp+10h] [rbp-300h]
  __int64 *v39; // [rsp+18h] [rbp-2F8h]
  int v40; // [rsp+20h] [rbp-2F0h]
  int v41; // [rsp+24h] [rbp-2ECh]
  __int64 v44[4]; // [rsp+40h] [rbp-2D0h] BYREF
  _QWORD v45[2]; // [rsp+60h] [rbp-2B0h] BYREF
  __int64 v46; // [rsp+70h] [rbp-2A0h]
  unsigned __int64 v47; // [rsp+78h] [rbp-298h]
  __int64 v48; // [rsp+80h] [rbp-290h] BYREF
  __int64 v49; // [rsp+88h] [rbp-288h]
  __int64 v50; // [rsp+90h] [rbp-280h]
  __int64 v51; // [rsp+98h] [rbp-278h]
  __int64 v52; // [rsp+A0h] [rbp-270h]
  unsigned __int64 v53; // [rsp+A8h] [rbp-268h]
  __int64 v54; // [rsp+B0h] [rbp-260h]
  __int64 v55; // [rsp+B8h] [rbp-258h]
  __int64 v56; // [rsp+C0h] [rbp-250h]
  __int64 v57; // [rsp+C8h] [rbp-248h]
  const char *v58; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v59; // [rsp+D8h] [rbp-238h]
  _BYTE v60[560]; // [rsp+E0h] [rbp-230h] BYREF

  if ( sub_15127D0(*(_QWORD *)(a2 + 232), 16, 0) )
  {
    v60[1] = 1;
    v58 = "Invalid record";
    v60[0] = 3;
    sub_1514BE0(a1, (__int64)&v58);
    return a1;
  }
  v7 = &v48;
  v59 = 0x4000000000LL;
  v8 = v45;
  v58 = v60;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  sub_1516C10(&v48, 0);
  while ( 1 )
  {
    do
    {
LABEL_5:
      v9 = sub_14ED070(*(_QWORD *)(a2 + 232), 0);
      if ( (_DWORD)v9 == 1 )
      {
        sub_1520420(a2, (__int64)v7, v10, v11, v12);
        *a1 = 1;
        goto LABEL_13;
      }
      if ( (v9 & 0xFFFFFFFD) == 0 )
      {
        BYTE1(v46) = 1;
        v19 = "Malformed block";
        goto LABEL_21;
      }
      v13 = *(_QWORD *)(a2 + 232);
      LODWORD(v59) = 0;
    }
    while ( (unsigned int)sub_1510D70(v13, SHIDWORD(v9), (__int64)&v58, 0) != 11 );
    if ( !(_DWORD)v59 )
      break;
    v14 = (__int64)v58;
    v41 = v59 & 1;
    if ( (v59 & 1) != 0 )
    {
      v37 = *(_QWORD *)(*a4 + 8LL * *(_QWORD *)v58);
      if ( (_DWORD)v59 == 1 )
        goto LABEL_5;
      v40 = v59;
      v39 = v8;
      v38 = (__int64)v7;
      v20 = v59 & 1;
      while ( 1 )
      {
        v21 = *(unsigned int *)(a2 + 1000);
        if ( !(_DWORD)v21 )
          goto LABEL_42;
        v22 = *(_QWORD *)(a2 + 984);
        v23 = *(_QWORD *)(v14 + 8LL * v20);
        v24 = (v21 - 1) & (37 * v23);
        v25 = (int *)(v22 + 8LL * v24);
        v26 = v41;
        v27 = *v25;
        if ( (_DWORD)v23 != *v25 )
        {
          while ( v27 != -1 )
          {
            v24 = (v21 - 1) & (v24 + v26);
            v25 = (int *)(v22 + 8LL * v24);
            v27 = *v25;
            if ( (unsigned int)*(_QWORD *)(v14 + 8LL * v20) == *v25 )
              goto LABEL_26;
            ++v26;
          }
LABEL_42:
          BYTE1(v46) = 1;
          v8 = v39;
          v19 = "Invalid ID";
          goto LABEL_21;
        }
LABEL_26:
        if ( v25 == (int *)(v22 + 8 * v21) )
          goto LABEL_42;
        if ( v25[1] != 1 || !*(_BYTE *)(a2 + 1008) )
        {
          v28 = *(_QWORD *)(v14 + 8LL * (v20 + 1));
          if ( ((__int64)(*(_QWORD *)(a2 + 664) - *(_QWORD *)(a2 + 656)) >> 3)
             + ((__int64)(*(_QWORD *)(a2 + 640) - *(_QWORD *)(a2 + 632)) >> 4) > v28
            && (*(_DWORD *)(a2 + 8) <= (unsigned int)v28 || !*(_QWORD *)(*(_QWORD *)a2 + 8LL * (unsigned int)v28)) )
          {
            v36 = v28;
            sub_15201C0(a2, v28, v38);
            sub_1520420(a2, v38, v31, v32, v33);
            LODWORD(v28) = v36;
          }
          v29 = (_BYTE *)sub_1517EB0(a2, v28);
          if ( *v29 == 2 )
          {
LABEL_49:
            v8 = v39;
            v7 = (__int64 *)v38;
            goto LABEL_5;
          }
          if ( (unsigned __int8)(*v29 - 4) > 0x1Eu )
          {
            BYTE1(v46) = 1;
            v8 = v39;
            v19 = "Invalid metadata attachment";
            goto LABEL_21;
          }
          v30 = (unsigned int)v25[1];
          if ( *(_BYTE *)(a2 + 1009) && (_DWORD)v30 == 18 )
          {
            v35 = sub_156A1F0(v29);
            v30 = (unsigned int)v25[1];
            v29 = (_BYTE *)v35;
          }
          if ( (_DWORD)v30 == 1 )
          {
            v34 = sub_1568EC0(v29);
            v30 = (unsigned int)v25[1];
            v29 = (_BYTE *)v34;
          }
          sub_1625C10(v37, v30, v29);
        }
        v20 += 2;
        if ( v20 == v40 )
          goto LABEL_49;
        v14 = (__int64)v58;
      }
    }
    sub_1518010(v8, a2, a3, (__int64)v58, v59);
    if ( (v45[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = v45[0] & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_13;
    }
  }
  BYTE1(v46) = 1;
  v19 = "Invalid record";
LABEL_21:
  v45[0] = v19;
  LOBYTE(v46) = 3;
  sub_1514BE0(a1, (__int64)v8);
LABEL_13:
  v44[0] = v54;
  v44[1] = v55;
  v44[2] = v56;
  v44[3] = v57;
  v45[0] = v50;
  v45[1] = v51;
  v46 = v52;
  v47 = v53;
  sub_1514A90(v8, v44);
  v15 = v48;
  if ( v48 )
  {
    v16 = (__int64 *)v53;
    v17 = v57 + 8;
    if ( v57 + 8 > v53 )
    {
      do
      {
        v18 = *v16++;
        j_j___libc_free_0(v18, 512);
      }
      while ( v17 > (unsigned __int64)v16 );
      v15 = v48;
    }
    j_j___libc_free_0(v15, 8 * v49);
  }
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  return a1;
}

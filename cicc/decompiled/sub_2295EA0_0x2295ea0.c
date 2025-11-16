// Function: sub_2295EA0
// Address: 0x2295ea0
//
__int64 __fastcall sub_2295EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // edx
  __int64 v13; // rdi
  int v14; // edx
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r14
  char *v18; // r9
  __int64 v19; // rsi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r15
  char *v23; // r15
  __int64 *v24; // r14
  __int64 *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // r8
  char v32; // al
  __int64 v33; // r9
  __int64 v34; // rcx
  unsigned __int64 v35; // rax
  int v36; // r15d
  __int64 v37; // r13
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 j; // rdx
  __int64 v41; // r13
  __int64 v42; // r15
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // rcx
  int v46; // eax
  int v47; // eax
  unsigned __int64 **v48; // r13
  unsigned __int64 **i; // r14
  int v50; // r14d
  int v51; // r9d
  __int64 v52; // [rsp-10h] [rbp-E0h]
  _BYTE *v53; // [rsp+20h] [rbp-B0h]
  _BYTE *v54; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v55; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v56; // [rsp+28h] [rbp-A8h]
  __int64 *v57; // [rsp+38h] [rbp-98h] BYREF
  _BYTE *v58; // [rsp+40h] [rbp-90h] BYREF
  __int64 v59; // [rsp+48h] [rbp-88h]
  _BYTE v60[32]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v61[2]; // [rsp+70h] [rbp-60h] BYREF
  _BYTE v62[80]; // [rsp+80h] [rbp-50h] BYREF

  v6 = 0;
  if ( *(_BYTE *)a2 > 0x1Cu && (unsigned __int8)(*(_BYTE *)a2 - 61) <= 1u )
    v6 = *(_QWORD *)(a2 - 32);
  v9 = 0;
  if ( *(_BYTE *)a3 > 0x1Cu && (unsigned __int8)(*(_BYTE *)a3 - 61) <= 1u )
    v9 = *(_QWORD *)(a3 - 32);
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_DWORD *)(v10 + 24);
  v13 = *(_QWORD *)(v10 + 8);
  if ( v12 )
  {
    v14 = v12 - 1;
    v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
    {
LABEL_9:
      v18 = (char *)v16[1];
    }
    else
    {
      v46 = 1;
      while ( v17 != -4096 )
      {
        v51 = v46 + 1;
        v15 = v14 & (v46 + v15);
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v11 == *v16 )
          goto LABEL_9;
        v46 = v51;
      }
      v18 = 0;
    }
    v19 = *(_QWORD *)(a3 + 40);
    v20 = v14 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v21 = (__int64 *)(v13 + 16LL * v20);
    v22 = *v21;
    if ( v19 == *v21 )
    {
LABEL_11:
      v23 = (char *)v21[1];
    }
    else
    {
      v47 = 1;
      while ( v22 != -4096 )
      {
        v50 = v47 + 1;
        v20 = v14 & (v47 + v20);
        v21 = (__int64 *)(v13 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == v19 )
          goto LABEL_11;
        v47 = v50;
      }
      v23 = 0;
    }
  }
  else
  {
    v18 = 0;
    v23 = 0;
  }
  v53 = (_BYTE *)a3;
  v54 = (_BYTE *)a2;
  v24 = sub_DDFBA0(*(_QWORD *)(a1 + 8), v6, v18);
  v25 = sub_DDFBA0(*(_QWORD *)(a1 + 8), v9, v23);
  v26 = sub_D97190(*(_QWORD *)(a1 + 8), (__int64)v24);
  v27 = *(_QWORD *)(a1 + 8);
  v28 = v26;
  if ( *(_WORD *)(v26 + 24) == 15 )
  {
    v29 = sub_D97190(v27, (__int64)v25);
    LODWORD(v30) = 0;
    if ( *(_WORD *)(v29 + 24) == 15 && v28 == v29 )
    {
      v58 = v60;
      v61[0] = (unsigned __int64)v62;
      v59 = 0x400000000LL;
      v61[1] = 0x400000000LL;
      v32 = sub_2292E00(a1, v54, v53, (__int64)v24, (__int64)v25, (__int64)&v58, (__int64)v61);
      v34 = v52;
      if ( v32
        || (v30 = (unsigned int)sub_2293160(a1, v54, v53, (__int64)v24, (__int64)v25, (__int64)&v58, (__int64)v61),
            (_BYTE)v30) )
      {
        v35 = *(unsigned int *)(a4 + 8);
        v36 = v59;
        if ( (int)v59 != v35 )
        {
          v37 = 48LL * (int)v59;
          if ( (int)v59 < v35 )
          {
            v48 = (unsigned __int64 **)(*(_QWORD *)a4 + v37);
            for ( i = (unsigned __int64 **)(*(_QWORD *)a4 + 48 * v35); v48 != i; sub_228BF40(i + 3) )
            {
              i -= 6;
              sub_228BF40(i + 5);
              sub_228BF40(i + 4);
            }
          }
          else
          {
            v38 = *(unsigned int *)(a4 + 12);
            if ( (int)v59 > v38 )
            {
              sub_2295D00(a4, (int)v59, v38, v34, v30, v33);
              v35 = *(unsigned int *)(a4 + 8);
            }
            v39 = *(_QWORD *)a4 + 48 * v35;
            for ( j = v37 + *(_QWORD *)a4; j != v39; v39 += 48 )
            {
              if ( v39 )
              {
                *(_QWORD *)v39 = 0;
                *(_QWORD *)(v39 + 8) = 0;
                *(_DWORD *)(v39 + 16) = 0;
                *(_QWORD *)(v39 + 24) = 1;
                *(_QWORD *)(v39 + 32) = 1;
                *(_QWORD *)(v39 + 40) = 1;
              }
            }
          }
          *(_DWORD *)(a4 + 8) = v36;
        }
        if ( v36 > 0 )
        {
          v41 = (unsigned int)v36;
          v42 = 0;
          v43 = 8 * v41;
          do
          {
            v44 = 6 * v42;
            *(_QWORD *)(*(_QWORD *)a4 + v44) = *(_QWORD *)&v58[v42];
            v45 = *(_QWORD *)(v61[0] + v42);
            v42 += 8;
            *(_QWORD *)(*(_QWORD *)a4 + v44 + 8) = v45;
            v57 = (__int64 *)(*(_QWORD *)a4 + v44);
            sub_228D890(a1, &v57, 1);
          }
          while ( v42 != v43 );
        }
        LODWORD(v30) = 1;
      }
      if ( (_BYTE *)v61[0] != v62 )
      {
        v55 = v30;
        _libc_free(v61[0]);
        LODWORD(v30) = v55;
      }
      if ( v58 != v60 )
      {
        v56 = v30;
        _libc_free((unsigned __int64)v58);
        LODWORD(v30) = v56;
      }
    }
  }
  else
  {
    sub_D97190(v27, (__int64)v25);
    LODWORD(v30) = 0;
  }
  return (unsigned int)v30;
}

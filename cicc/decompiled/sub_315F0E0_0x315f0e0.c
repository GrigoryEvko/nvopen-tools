// Function: sub_315F0E0
// Address: 0x315f0e0
//
__int64 __fastcall sub_315F0E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 **a5)
{
  __int64 i; // r14
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rdx
  bool v13; // cc
  unsigned __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // r15
  __int64 j; // r13
  __int64 v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rbx
  __int64 v33; // r12
  unsigned __int64 v34; // rdi
  __int64 v36; // r14
  __int64 v37; // r12
  unsigned int v38; // edx
  __int64 v39; // r13
  __int64 v40; // rbx
  unsigned __int64 v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // [rsp+8h] [rbp-98h]
  int v45; // [rsp+14h] [rbp-8Ch]
  __int64 v47; // [rsp+18h] [rbp-88h]
  char v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+28h] [rbp-78h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  __int64 v53; // [rsp+38h] [rbp-68h]
  __int64 v54; // [rsp+38h] [rbp-68h]
  unsigned __int64 v55; // [rsp+40h] [rbp-60h]
  __int64 v56; // [rsp+40h] [rbp-60h]
  __int64 v57; // [rsp+40h] [rbp-60h]
  unsigned __int64 v58; // [rsp+40h] [rbp-60h]
  __int64 v59; // [rsp+40h] [rbp-60h]
  __int64 **v60; // [rsp+40h] [rbp-60h]
  _QWORD v62[10]; // [rsp+50h] [rbp-50h] BYREF

  v50 = (a3 - 1) / 2;
  if ( a2 >= v50 )
  {
    v17 = a2;
    v8 = a1 + 48 * a2;
  }
  else
  {
    for ( i = a2; ; i = v7 )
    {
      v7 = 2 * (i + 1);
      v8 = a1 + 96 * (i + 1);
      sub_B4CED0((__int64)v62, *(_QWORD *)v8, **a5);
      v55 = v62[0];
      sub_B4CED0((__int64)v62, *(_QWORD *)(v8 - 48), **a5);
      if ( v62[0] < v55 )
      {
        --v7;
        v8 = a1 + 48 * v7;
      }
      v9 = a1 + 48 * i;
      v10 = *(unsigned int *)(v9 + 32);
      *(_QWORD *)v9 = *(_QWORD *)v8;
      if ( (_DWORD)v10 )
      {
        v11 = *(_QWORD *)(v9 + 16);
        v12 = v11 + 32 * v10;
        do
        {
          while ( 1 )
          {
            if ( *(_QWORD *)v11 != -4096 && *(_QWORD *)v11 != -8192 )
            {
              if ( *(_BYTE *)(v11 + 24) )
              {
                v13 = *(_DWORD *)(v11 + 16) <= 0x40u;
                *(_BYTE *)(v11 + 24) = 0;
                if ( !v13 )
                {
                  v14 = *(_QWORD *)(v11 + 8);
                  if ( v14 )
                    break;
                }
              }
            }
            v11 += 32;
            if ( v12 == v11 )
              goto LABEL_14;
          }
          v53 = v12;
          v11 += 32;
          v56 = v9;
          j_j___libc_free_0_0(v14);
          v12 = v53;
          v9 = v56;
        }
        while ( v53 != v11 );
LABEL_14:
        v10 = *(unsigned int *)(v9 + 32);
      }
      v57 = v9;
      sub_C7D6A0(*(_QWORD *)(v9 + 16), 32 * v10, 8);
      *(_QWORD *)(v57 + 24) = 0;
      *(_QWORD *)(v57 + 16) = 0;
      *(_DWORD *)(v57 + 32) = 0;
      ++*(_QWORD *)(v57 + 8);
      v15 = *(_QWORD *)(v8 + 16);
      ++*(_QWORD *)(v8 + 8);
      v16 = *(_QWORD *)(v57 + 16);
      *(_QWORD *)(v57 + 16) = v15;
      LODWORD(v15) = *(_DWORD *)(v8 + 24);
      *(_QWORD *)(v8 + 16) = v16;
      LODWORD(v16) = *(_DWORD *)(v57 + 24);
      *(_DWORD *)(v57 + 24) = v15;
      LODWORD(v15) = *(_DWORD *)(v8 + 28);
      *(_DWORD *)(v8 + 24) = v16;
      LODWORD(v16) = *(_DWORD *)(v57 + 28);
      *(_DWORD *)(v57 + 28) = v15;
      LODWORD(v15) = *(_DWORD *)(v8 + 32);
      *(_DWORD *)(v8 + 28) = v16;
      LODWORD(v16) = *(_DWORD *)(v57 + 32);
      *(_DWORD *)(v57 + 32) = v15;
      *(_DWORD *)(v8 + 32) = v16;
      *(_BYTE *)(v57 + 40) = *(_BYTE *)(v8 + 40);
      if ( v7 >= v50 )
        break;
    }
    v17 = v7;
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v17 )
  {
    v36 = 2 * (v17 + 1);
    v37 = a1 + 96 * (v17 + 1) - 48;
    *(_QWORD *)v8 = *(_QWORD *)v37;
    v38 = *(_DWORD *)(v8 + 32);
    if ( v38 )
    {
      v39 = *(_QWORD *)(v8 + 16);
      v60 = a5;
      v40 = v39 + 32LL * v38;
      do
      {
        if ( *(_QWORD *)v39 != -8192 && *(_QWORD *)v39 != -4096 )
        {
          if ( *(_BYTE *)(v39 + 24) )
          {
            v13 = *(_DWORD *)(v39 + 16) <= 0x40u;
            *(_BYTE *)(v39 + 24) = 0;
            if ( !v13 )
            {
              v41 = *(_QWORD *)(v39 + 8);
              if ( v41 )
                j_j___libc_free_0_0(v41);
            }
          }
        }
        v39 += 32;
      }
      while ( v40 != v39 );
      a5 = v60;
      v38 = *(_DWORD *)(v8 + 32);
    }
    v17 = v36 - 1;
    sub_C7D6A0(*(_QWORD *)(v8 + 16), 32LL * v38, 8);
    ++*(_QWORD *)(v8 + 8);
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_DWORD *)(v8 + 32) = 0;
    v42 = *(_QWORD *)(v37 + 16);
    ++*(_QWORD *)(v37 + 8);
    v43 = *(_QWORD *)(v8 + 16);
    *(_QWORD *)(v8 + 16) = v42;
    LODWORD(v42) = *(_DWORD *)(v37 + 24);
    *(_QWORD *)(v37 + 16) = v43;
    LODWORD(v43) = *(_DWORD *)(v8 + 24);
    *(_DWORD *)(v8 + 24) = v42;
    LODWORD(v42) = *(_DWORD *)(v37 + 28);
    *(_DWORD *)(v37 + 24) = v43;
    LODWORD(v43) = *(_DWORD *)(v8 + 28);
    *(_DWORD *)(v8 + 28) = v42;
    LODWORD(v42) = *(_DWORD *)(v37 + 32);
    *(_DWORD *)(v37 + 28) = v43;
    LODWORD(v43) = *(_DWORD *)(v8 + 32);
    *(_DWORD *)(v8 + 32) = v42;
    *(_DWORD *)(v37 + 32) = v43;
    *(_BYTE *)(v8 + 40) = *(_BYTE *)(v37 + 40);
    v8 = a1 + 48 * (v36 - 1);
  }
  v18 = a4;
  v19 = *a4;
  ++a4[1];
  v54 = v19;
  v20 = a4[2];
  a4[2] = 0;
  v47 = v20;
  LODWORD(v20) = *((_DWORD *)v18 + 8);
  *((_DWORD *)v18 + 8) = 0;
  v45 = v20;
  v21 = v18[3];
  v18[3] = 0;
  v44 = v21;
  v49 = *((_BYTE *)v18 + 40);
  if ( v17 > a2 )
  {
    v22 = v17;
    for ( j = (v17 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v24 = a1 + 48 * j;
      sub_B4CED0((__int64)v62, *(_QWORD *)v24, **a5);
      v58 = v62[0];
      sub_B4CED0((__int64)v62, v54, **a5);
      v8 = a1 + 48 * v22;
      if ( v62[0] >= v58 )
        break;
      v25 = *(unsigned int *)(v8 + 32);
      *(_QWORD *)v8 = *(_QWORD *)v24;
      if ( (_DWORD)v25 )
      {
        v26 = *(_QWORD *)(v8 + 16);
        v27 = v26 + 32 * v25;
        do
        {
          while ( 1 )
          {
            if ( *(_QWORD *)v26 != -8192 && *(_QWORD *)v26 != -4096 )
            {
              if ( *(_BYTE *)(v26 + 24) )
              {
                v13 = *(_DWORD *)(v26 + 16) <= 0x40u;
                *(_BYTE *)(v26 + 24) = 0;
                if ( !v13 )
                {
                  v28 = *(_QWORD *)(v26 + 8);
                  if ( v28 )
                    break;
                }
              }
            }
            v26 += 32;
            if ( v27 == v26 )
              goto LABEL_32;
          }
          v51 = v26;
          v59 = v27;
          j_j___libc_free_0_0(v28);
          v27 = v59;
          v26 = v51 + 32;
        }
        while ( v59 != v51 + 32 );
LABEL_32:
        v25 = *(unsigned int *)(v8 + 32);
      }
      sub_C7D6A0(*(_QWORD *)(v8 + 16), 32 * v25, 8);
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)(v8 + 16) = 0;
      *(_DWORD *)(v8 + 32) = 0;
      ++*(_QWORD *)(v8 + 8);
      v29 = *(_QWORD *)(v24 + 16);
      ++*(_QWORD *)(v24 + 8);
      v30 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v8 + 16) = v29;
      LODWORD(v29) = *(_DWORD *)(v24 + 24);
      *(_QWORD *)(v24 + 16) = v30;
      LODWORD(v30) = *(_DWORD *)(v8 + 24);
      *(_DWORD *)(v8 + 24) = v29;
      LODWORD(v29) = *(_DWORD *)(v24 + 28);
      *(_DWORD *)(v24 + 24) = v30;
      LODWORD(v30) = *(_DWORD *)(v8 + 28);
      *(_DWORD *)(v8 + 28) = v29;
      LODWORD(v29) = *(_DWORD *)(v24 + 32);
      *(_DWORD *)(v24 + 28) = v30;
      LODWORD(v30) = *(_DWORD *)(v8 + 32);
      *(_DWORD *)(v8 + 32) = v29;
      *(_DWORD *)(v24 + 32) = v30;
      *(_BYTE *)(v8 + 40) = *(_BYTE *)(v24 + 40);
      v22 = j;
      if ( a2 >= j )
      {
        v8 = a1 + 48 * j;
        break;
      }
    }
  }
  v31 = *(unsigned int *)(v8 + 32);
  *(_QWORD *)v8 = v54;
  if ( (_DWORD)v31 )
  {
    v32 = *(_QWORD *)(v8 + 16);
    v33 = v32 + 32 * v31;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v32 != -8192 && *(_QWORD *)v32 != -4096 )
        {
          if ( *(_BYTE *)(v32 + 24) )
          {
            v13 = *(_DWORD *)(v32 + 16) <= 0x40u;
            *(_BYTE *)(v32 + 24) = 0;
            if ( !v13 )
            {
              v34 = *(_QWORD *)(v32 + 8);
              if ( v34 )
                break;
            }
          }
        }
        v32 += 32;
        if ( v33 == v32 )
          goto LABEL_45;
      }
      j_j___libc_free_0_0(v34);
      v32 += 32;
    }
    while ( v33 != v32 );
LABEL_45:
    LODWORD(v31) = *(_DWORD *)(v8 + 32);
  }
  sub_C7D6A0(*(_QWORD *)(v8 + 16), 32LL * (unsigned int)v31, 8);
  ++*(_QWORD *)(v8 + 8);
  *(_QWORD *)(v8 + 16) = v47;
  *(_QWORD *)(v8 + 24) = v44;
  *(_DWORD *)(v8 + 32) = v45;
  *(_BYTE *)(v8 + 40) = v49;
  return sub_C7D6A0(0, 0, 8);
}

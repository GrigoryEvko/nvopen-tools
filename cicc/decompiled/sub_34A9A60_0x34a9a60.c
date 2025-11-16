// Function: sub_34A9A60
// Address: 0x34a9a60
//
__int64 __fastcall sub_34A9A60(__int64 *a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  __int64 v7; // rcx
  unsigned int *v8; // rax
  __int64 v9; // rsi
  char v10; // cl
  __int64 v11; // rdi
  int v12; // edx
  unsigned int v13; // r8d
  int *v14; // rax
  int v15; // r9d
  _QWORD *v16; // rbx
  __int64 v17; // r13
  int v18; // r15d
  int v19; // eax
  int v20; // edi
  unsigned int i; // eax
  __int64 v22; // r14
  unsigned int v23; // eax
  __int64 v24; // r15
  __int64 v25; // rbx
  __int64 v26; // r14
  char v27; // al
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r9
  unsigned int *v33; // r11
  unsigned int *v34; // rbx
  unsigned int *v35; // r13
  unsigned __int64 v36; // r8
  __int64 v37; // rax
  unsigned int *v38; // rdx
  unsigned __int64 v39; // rcx
  unsigned int *v40; // rdx
  unsigned __int64 v41; // rsi
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  unsigned int v44; // eax
  __int64 v45; // rax
  int v46; // eax
  int v47; // r10d
  __int64 v48; // [rsp+0h] [rbp-1E0h]
  int v49; // [rsp+Ch] [rbp-1D4h]
  __int64 v51; // [rsp+18h] [rbp-1C8h]
  __int64 v52; // [rsp+28h] [rbp-1B8h]
  char v54; // [rsp+37h] [rbp-1A9h]
  __int64 v55; // [rsp+38h] [rbp-1A8h]
  int v56; // [rsp+4Ch] [rbp-194h] BYREF
  unsigned int *v57; // [rsp+50h] [rbp-190h] BYREF
  __int64 v58; // [rsp+58h] [rbp-188h]
  _BYTE v59[16]; // [rsp+60h] [rbp-180h] BYREF
  unsigned int *v60; // [rsp+70h] [rbp-170h] BYREF
  _QWORD *v61; // [rsp+78h] [rbp-168h]
  __int64 v62; // [rsp+80h] [rbp-160h]
  _QWORD v63[9]; // [rsp+88h] [rbp-158h] BYREF
  __int64 v64; // [rsp+D0h] [rbp-110h] BYREF
  unsigned int v65[48]; // [rsp+D8h] [rbp-108h] BYREF
  __int64 v66; // [rsp+198h] [rbp-48h]
  __int64 v67; // [rsp+1A0h] [rbp-40h]

  v64 = *a1;
  v67 = v64;
  v6 = *(_QWORD *)(a2 + 184) == 0;
  memset(v65, 0, sizeof(v65));
  v7 = 0;
  v55 = a3;
  v66 = 0;
  if ( v6 )
  {
    v7 = *(_QWORD *)a2;
    v54 = 1;
    v52 = *(_QWORD *)a2;
    v48 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  }
  else
  {
    v54 = 0;
    v52 = *(_QWORD *)(a2 + 168);
    v48 = a2 + 152;
  }
  v49 = 37 * a4;
  if ( v54 )
    goto LABEL_31;
LABEL_4:
  if ( v48 != v52 )
  {
    v8 = (unsigned int *)(v52 + 32);
    while ( 1 )
    {
      v9 = *v8;
      v10 = *(_BYTE *)(v55 + 56) & 1;
      if ( v10 )
      {
        v11 = v55 + 64;
        v12 = 3;
      }
      else
      {
        v11 = *(_QWORD *)(v55 + 64);
        v42 = *(unsigned int *)(v55 + 72);
        if ( !(_DWORD)v42 )
          goto LABEL_63;
        v12 = v42 - 1;
      }
      v13 = v12 & v49;
      v14 = (int *)(v11 + 32LL * (v12 & (unsigned int)v49));
      v15 = *v14;
      if ( a4 != *v14 )
        break;
LABEL_9:
      a5 = *((_QWORD *)v14 + 1) + 384 * v9;
      a3 = (__int64)(a1 + 102);
      v16 = a1 + 28;
      v17 = a5;
      if ( (unsigned int)(*(_DWORD *)(a5 + 56) - 2) <= 1 )
        v16 = a1 + 102;
      if ( (v16[1] & 1) != 0 )
      {
        a6 = (__int64)(v16 + 2);
        v18 = 7;
      }
      else
      {
        v7 = *((unsigned int *)v16 + 6);
        a6 = v16[2];
        v18 = v7 - 1;
        if ( !(_DWORD)v7 )
          goto LABEL_20;
      }
      v56 = 0;
      if ( *(_BYTE *)(a5 + 24) )
        v56 = *(unsigned __int16 *)(a5 + 16) | (*(_DWORD *)(a5 + 8) << 16);
      v51 = a6;
      v60 = *(unsigned int **)(a5 + 32);
      v57 = *(unsigned int **)a5;
      v19 = sub_F11290((__int64 *)&v57, &v56, (__int64 *)&v60);
      v20 = 1;
      a6 = v51;
      for ( i = v18 & v19; ; i = v18 & v23 )
      {
        v22 = v51 + 72LL * i;
        a3 = *(_QWORD *)v22;
        if ( *(_QWORD *)v22 == *(_QWORD *)v17 )
        {
          v7 = *(unsigned __int8 *)(v17 + 24);
          if ( (_BYTE)v7 == *(_BYTE *)(v22 + 24) )
          {
            if ( (_BYTE)v7 )
            {
              v7 = *(_QWORD *)(v22 + 8);
              if ( *(_QWORD *)(v17 + 8) != v7 )
                goto LABEL_18;
              v7 = *(_QWORD *)(v22 + 16);
              if ( *(_QWORD *)(v17 + 16) != v7 )
                goto LABEL_18;
            }
            v7 = *(_QWORD *)(v22 + 32);
            if ( *(_QWORD *)(v17 + 32) == v7 )
              break;
          }
        }
        if ( !a3 && !*(_BYTE *)(v22 + 24) && !*(_QWORD *)(v22 + 32) )
          goto LABEL_20;
LABEL_18:
        v23 = v20 + i;
        ++v20;
      }
      v43 = *(_QWORD *)(v22 + 40);
      if ( v43 != v22 + 56 )
        _libc_free(v43);
      *(_QWORD *)v22 = 0;
      *(_QWORD *)(v22 + 8) = 0;
      *(_QWORD *)(v22 + 16) = 0;
      *(_BYTE *)(v22 + 24) = 1;
      *(_QWORD *)(v22 + 32) = 0;
      v44 = *((_DWORD *)v16 + 2);
      ++*((_DWORD *)v16 + 3);
      a3 = 2 * (v44 >> 1) - 2;
      *((_DWORD *)v16 + 2) = a3 | v44 & 1;
LABEL_20:
      v24 = *(_QWORD *)(v55 + 16);
      v25 = v55 + 8;
      if ( v24 )
      {
        v26 = v55 + 8;
        do
        {
          while ( 1 )
          {
            v27 = sub_34A0190(v24 + 32, v17);
            a3 = *(_QWORD *)(v24 + 16);
            v7 = *(_QWORD *)(v24 + 24);
            if ( v27 )
              break;
            v26 = v24;
            v24 = *(_QWORD *)(v24 + 16);
            if ( !a3 )
              goto LABEL_25;
          }
          v24 = *(_QWORD *)(v24 + 24);
        }
        while ( v7 );
LABEL_25:
        if ( v26 != v25 && !(unsigned __int8)sub_34A0190(v17, v26 + 32) )
          v25 = v26;
      }
      v57 = (unsigned int *)v59;
      v58 = 0x200000000LL;
      if ( !*(_DWORD *)(v25 + 424) )
      {
LABEL_29:
        if ( !v54 )
          goto LABEL_46;
        goto LABEL_30;
      }
      sub_349DD80((__int64)&v57, v25 + 416, a3, v7, a5, a6);
      v33 = v57;
      v34 = &v57[2 * (unsigned int)v58];
      if ( v34 == v57 )
        goto LABEL_44;
      v35 = v57;
      while ( 2 )
      {
        v36 = v35[1] | ((unsigned __int64)*v35 << 32);
        v37 = (unsigned int)v66;
        if ( (_DWORD)v66 )
        {
          v41 = v35[1] | ((unsigned __int64)*v35 << 32);
          v60 = v65;
          v61 = v63;
          v62 = 0x400000000LL;
          sub_34A3C90((__int64)&v60, v41, a3, v7, v36, a6);
          v36 = v41;
        }
        else
        {
          if ( HIDWORD(v66) != 11 )
          {
            if ( HIDWORD(v66) )
            {
              v38 = &v65[2];
              do
              {
                if ( v36 <= *(_QWORD *)v38 )
                  break;
                LODWORD(v37) = v37 + 1;
                v38 += 4;
              }
              while ( HIDWORD(v66) != (_DWORD)v37 );
            }
            v39 = v35[1] | ((unsigned __int64)*v35 << 32);
            LODWORD(v60) = v37;
            HIDWORD(v66) = sub_34A32D0((__int64)v65, (unsigned int *)&v60, HIDWORD(v66), v39, v36, 0);
            goto LABEL_42;
          }
          v60 = v65;
          v40 = &v65[2];
          v61 = v63;
          HIDWORD(v62) = 4;
          do
          {
            if ( v36 <= *(_QWORD *)v40 )
              break;
            v37 = (unsigned int)(v37 + 1);
            v40 += 4;
          }
          while ( (_DWORD)v37 != 11 );
          v63[0] = v65;
          LODWORD(v62) = 1;
          v63[1] = (v37 << 32) | 0xB;
        }
        sub_34A8E00((__int64)&v60, v36, v36, 0);
        if ( v61 != v63 )
          _libc_free((unsigned __int64)v61);
LABEL_42:
        v35 += 2;
        if ( v34 != v35 )
          continue;
        break;
      }
      v33 = v57;
LABEL_44:
      if ( v33 == (unsigned int *)v59 )
        goto LABEL_29;
      _libc_free((unsigned __int64)v33);
      if ( !v54 )
      {
LABEL_46:
        v52 = sub_220EF30(v52);
        goto LABEL_4;
      }
LABEL_30:
      v52 += 4;
LABEL_31:
      v8 = (unsigned int *)v52;
      if ( v48 == v52 )
        goto LABEL_32;
    }
    v46 = 1;
    while ( v15 != -1 )
    {
      v47 = v46 + 1;
      v13 = v12 & (v46 + v13);
      v14 = (int *)(v11 + 32LL * v13);
      v15 = *v14;
      if ( a4 == *v14 )
        goto LABEL_9;
      v46 = v47;
    }
    if ( v10 )
    {
      v45 = 128;
    }
    else
    {
      v42 = *(unsigned int *)(v55 + 72);
LABEL_63:
      v45 = 32 * v42;
    }
    v14 = (int *)(v11 + v45);
    goto LABEL_9;
  }
LABEL_32:
  sub_34A9020((__int64)(a1 + 1), (__int64)&v64, a3, v7, a5, a6);
  return sub_34A2530(v65, (__int64)&v64, v28, v29, v30, v31);
}

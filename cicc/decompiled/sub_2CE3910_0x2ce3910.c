// Function: sub_2CE3910
// Address: 0x2ce3910
//
void __fastcall sub_2CE3910(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  __int64 i; // rbx
  __int64 v6; // r9
  _BYTE *v7; // rsi
  unsigned __int64 *v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // rax
  int v13; // eax
  _BYTE *v14; // rax
  _BYTE *v15; // rcx
  unsigned __int64 *v16; // rax
  unsigned __int64 *v17; // r9
  unsigned __int64 *v18; // rdx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  unsigned __int64 *v21; // rsi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // rsi
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  _QWORD *v28; // r9
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rsi
  char v33; // al
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r12
  _QWORD *v38; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+20h] [rbp-C0h]
  __int64 v41; // [rsp+28h] [rbp-B8h]
  __int64 v42; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v43; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v44; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v45; // [rsp+48h] [rbp-98h]
  _BYTE *v46; // [rsp+50h] [rbp-90h]
  unsigned __int64 *v47[4]; // [rsp+60h] [rbp-80h] BYREF
  char v48[8]; // [rsp+80h] [rbp-60h] BYREF
  int v49; // [rsp+88h] [rbp-58h] BYREF
  unsigned __int64 v50; // [rsp+90h] [rbp-50h]
  int *v51; // [rsp+98h] [rbp-48h]
  int *v52; // [rsp+A0h] [rbp-40h]
  __int64 v53; // [rsp+A8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  v49 = 0;
  v50 = 0;
  v51 = &v49;
  v52 = &v49;
  v53 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  for ( i = v2 + 24; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    if ( !v3 )
      BUG();
    if ( *(_BYTE *)(v3 - 24) == 60 && (unsigned __int8)sub_2CDDA20(v3 - 24) )
    {
      v47[0] = (unsigned __int64 *)(v3 - 24);
      v47[1] = (unsigned __int64 *)(v3 - 24);
      v47[2] = 0;
      sub_2CE1820((__int64)v48, (__int64 *)v47);
      sub_2CE1620(a1, v3 - 24, v3 - 24, 0, (__int64)v48, v6);
      v47[0] = (unsigned __int64 *)(v3 - 24);
      v7 = v45;
      if ( v45 == v46 )
      {
        sub_928380((__int64)&v44, v45, v47);
      }
      else
      {
        if ( v45 )
        {
          *(_QWORD *)v45 = v3 - 24;
          v7 = v45;
        }
        v45 = v7 + 8;
      }
    }
  }
  v8 = (unsigned __int64 *)(a1 + 384);
  v38 = (_QWORD *)(a1 + 376);
  sub_2CDEE00(*(_QWORD **)(a1 + 392));
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = a1 + 384;
  *(_QWORD *)(a1 + 408) = a1 + 384;
  *(_QWORD *)(a1 + 416) = 0;
  v9 = *(_QWORD *)(a2 + 80);
  v40 = a2 + 72;
  if ( v9 != v40 )
  {
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      v10 = v9 + 24;
      if ( v9 + 24 != *(_QWORD *)(v9 + 32) )
        break;
LABEL_55:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v40 == v9 )
        goto LABEL_56;
    }
    v41 = v9;
    v11 = *(_QWORD *)(v9 + 32);
    while ( 1 )
    {
      if ( !v11 )
        BUG();
      if ( *(_BYTE *)(v11 - 24) != 85 )
        goto LABEL_15;
      v12 = *(_QWORD *)(v11 - 56);
      if ( !v12 )
        goto LABEL_15;
      if ( *(_BYTE *)v12 )
        goto LABEL_15;
      if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v11 + 56) )
        goto LABEL_15;
      if ( (*(_BYTE *)(v12 + 33) & 0x20) == 0 )
        goto LABEL_15;
      v13 = *(_DWORD *)(v12 + 36);
      if ( v13 != 68 && v13 != 71 )
        goto LABEL_15;
      v42 = v11 - 24;
      v14 = (_BYTE *)sub_B58EB0(v11 - 24, 0);
      v15 = v14;
      if ( !v14 || *v14 != 61 )
        goto LABEL_15;
      v16 = *(unsigned __int64 **)(a1 + 392);
      v43 = (unsigned __int64)v15;
      if ( !v16 )
        break;
      v17 = (unsigned __int64 *)(a1 + 384);
      v18 = v16;
      do
      {
        while ( 1 )
        {
          v19 = v18[2];
          v20 = v18[3];
          if ( v18[4] >= (unsigned __int64)v15 )
            break;
          v18 = (unsigned __int64 *)v18[3];
          if ( !v20 )
            goto LABEL_31;
        }
        v17 = v18;
        v18 = (unsigned __int64 *)v18[2];
      }
      while ( v19 );
LABEL_31:
      if ( v8 != v17 && v17[4] <= (unsigned __int64)v15 )
      {
LABEL_43:
        v25 = (unsigned __int64 *)(a1 + 384);
        do
        {
          while ( 1 )
          {
            v26 = v16[2];
            v27 = v16[3];
            if ( v16[4] >= v43 )
              break;
            v16 = (unsigned __int64 *)v16[3];
            if ( !v27 )
              goto LABEL_47;
          }
          v25 = v16;
          v16 = (unsigned __int64 *)v16[2];
        }
        while ( v26 );
LABEL_47:
        if ( v8 != v25 && v25[4] <= v43 )
          goto LABEL_50;
        goto LABEL_49;
      }
      v21 = (unsigned __int64 *)(a1 + 384);
      do
      {
        while ( 1 )
        {
          v22 = v16[2];
          v23 = v16[3];
          if ( v16[4] >= (unsigned __int64)v15 )
            break;
          v16 = (unsigned __int64 *)v16[3];
          if ( !v23 )
            goto LABEL_37;
        }
        v21 = v16;
        v16 = (unsigned __int64 *)v16[2];
      }
      while ( v22 );
LABEL_37:
      if ( v8 == v21 || v21[4] > (unsigned __int64)v15 )
        goto LABEL_39;
LABEL_40:
      v24 = v21[5];
      v21[5] = 0;
      v21[6] = 0;
      v21[7] = 0;
      if ( v24 )
        j_j___libc_free_0(v24);
      v16 = *(unsigned __int64 **)(a1 + 392);
      if ( v16 )
        goto LABEL_43;
      v25 = (unsigned __int64 *)(a1 + 384);
LABEL_49:
      v47[0] = &v43;
      v25 = sub_2CE35C0(v38, (__int64)v25, v47);
LABEL_50:
      v28 = (_QWORD *)v25[6];
      if ( v28 == (_QWORD *)v25[7] )
      {
        sub_2CE0C80((__int64)(v25 + 5), (_BYTE *)v25[6], &v42);
LABEL_15:
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
          goto LABEL_54;
      }
      else
      {
        if ( v28 )
        {
          *v28 = v42;
          v28 = (_QWORD *)v25[6];
        }
        v25[6] = (unsigned __int64)(v28 + 1);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 == v11 )
        {
LABEL_54:
          v9 = v41;
          goto LABEL_55;
        }
      }
    }
    v21 = (unsigned __int64 *)(a1 + 384);
LABEL_39:
    v47[0] = &v43;
    v21 = sub_2CE35C0(v38, (__int64)v21, v47);
    goto LABEL_40;
  }
LABEL_56:
  v29 = *(_QWORD *)(a1 + 424);
  v30 = *(_QWORD *)(a1 + 432);
  v31 = (__int64)v51;
  if ( v29 == v30 )
  {
    if ( v51 != &v49 )
    {
      do
      {
LABEL_58:
        v32 = *(_QWORD *)(v31 + 32);
        if ( *(_BYTE *)(*(_QWORD *)(v32 + 8) + 8LL) == 14 )
        {
          v33 = *(_BYTE *)v32;
          if ( *(_BYTE *)v32 <= 0x1Cu )
            goto LABEL_81;
          if ( v33 == 63 )
          {
            v34 = *(_QWORD *)(v32 + 80);
          }
          else
          {
            if ( v33 != 60 )
LABEL_81:
              BUG();
            v34 = *(_QWORD *)(v32 + 72);
          }
          if ( *(_BYTE *)(v34 + 8) == 15 )
            sub_2CE3810((_QWORD *)a1, v32);
        }
        v31 = sub_220EEE0(v31);
      }
      while ( (int *)v31 != &v49 );
      v30 = *(_QWORD *)(a1 + 432);
      v29 = *(_QWORD *)(a1 + 424);
    }
    v35 = (v30 - v29) >> 3;
    if ( (_DWORD)v35 )
    {
      v36 = 0;
      v37 = 8LL * (unsigned int)(v35 - 1);
      while ( 1 )
      {
        sub_B43D60(*(_QWORD **)(v29 + v36));
        if ( v36 == v37 )
          break;
        v29 = *(_QWORD *)(a1 + 424);
        v36 += 8;
      }
    }
  }
  else
  {
    *(_QWORD *)(a1 + 432) = v29;
    if ( (int *)v31 != &v49 )
      goto LABEL_58;
  }
  if ( v44 )
    j_j___libc_free_0(v44);
  sub_2CDEC30(v50);
}

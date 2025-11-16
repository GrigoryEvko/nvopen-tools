// Function: sub_31C1C20
// Address: 0x31c1c20
//
__int64 __fastcall sub_31C1C20(__int64 *a1, __int64 *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  _BYTE *v9; // r9
  char *v10; // rax
  char *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 *v14; // r14
  _QWORD *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 *v20; // r12
  __int64 *v21; // r14
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  int v24; // edx
  __int64 v25; // rax
  _BYTE *v26; // rsi
  __int64 v27; // r9
  __int64 v28; // rax
  _QWORD *v29; // rax
  unsigned int v30; // r12d
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rax
  char v46; // [rsp+7h] [rbp-A9h]
  _BYTE *v47; // [rsp+10h] [rbp-A0h]
  __int64 v48; // [rsp+20h] [rbp-90h]
  unsigned __int64 v49; // [rsp+28h] [rbp-88h]
  __int64 v50; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v51; // [rsp+40h] [rbp-70h] BYREF
  __int64 v52; // [rsp+48h] [rbp-68h]
  _BYTE v53[96]; // [rsp+50h] [rbp-60h] BYREF

  v48 = sub_31C1720((__int64)a1, a2, a3, a4, a5, a6);
  v51 = v53;
  v52 = 0x600000000LL;
  while ( 2 )
  {
    while ( 1 )
    {
      v10 = (char *)a1[2];
      v11 = (char *)a1[1];
      if ( v10 != v11 )
        break;
LABEL_12:
      v47 = &v51[8 * (unsigned int)v52];
      if ( v51 == v47 )
        goto LABEL_34;
      v49 = (unsigned __int64)v51;
      v46 = 0;
      do
      {
        v18 = *(_QWORD *)v49;
        v19 = *(_QWORD *)(*(_QWORD *)v49 + 32LL);
        if ( !v19 )
        {
          v50 = *(_QWORD *)(v18 + 8);
          v40 = sub_31C1720((__int64)a1, &v50, 1u, v7, v8, (__int64)v9);
          sub_31C05A0((__int64)a1, v40, v41, v42, v43, v44);
LABEL_21:
          v23 = v51;
          v24 = v52;
          v25 = 8LL * (unsigned int)v52;
          v26 = &v51[v25];
          v27 = v25 >> 3;
          v28 = v25 >> 5;
          if ( v28 )
          {
            v29 = &v51[32 * v28];
            while ( v18 != *v23 )
            {
              if ( v18 == v23[1] )
              {
                v9 = ++v23 + 1;
                goto LABEL_29;
              }
              if ( v18 == v23[2] )
              {
                v23 += 2;
                v9 = v23 + 1;
                goto LABEL_29;
              }
              if ( v18 == v23[3] )
              {
                v23 += 3;
                break;
              }
              v23 += 4;
              if ( v29 == v23 )
              {
                v27 = (v26 - (_BYTE *)v23) >> 3;
                goto LABEL_46;
              }
            }
LABEL_28:
            v9 = v23 + 1;
LABEL_29:
            if ( v9 != v26 )
            {
              memmove(v23, v9, v26 - v9);
              v24 = v52;
            }
            v46 = 1;
            LODWORD(v52) = v24 - 1;
            goto LABEL_32;
          }
LABEL_46:
          switch ( v27 )
          {
            case 2LL:
              v45 = v23;
              break;
            case 3LL:
              v9 = v23 + 1;
              v45 = v23 + 1;
              if ( v18 == *v23 )
                goto LABEL_29;
              break;
            case 1LL:
LABEL_53:
              if ( v18 == *v23 )
                goto LABEL_28;
              goto LABEL_49;
            default:
LABEL_49:
              v23 = v26;
              v9 = v26 + 8;
              goto LABEL_29;
          }
          v23 = v45 + 1;
          if ( v18 == *v45 )
          {
            v23 = v45;
            v9 = v45 + 1;
            goto LABEL_29;
          }
          goto LABEL_53;
        }
        v20 = *(__int64 **)v19;
        v21 = (__int64 *)(*(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 8));
        if ( v21 == sub_31BFDD0(*(_QWORD **)v19, (__int64)v21) )
        {
          for ( ; v21 != v20; ++v20 )
          {
            if ( v18 != *v20 )
              sub_31C1020(a1, *v20, v22, v7, v8, (__int64)v9);
          }
          sub_31C05A0((__int64)a1, v19, v22, v7, v8, (__int64)v9);
          if ( v48 != v19 )
            goto LABEL_21;
        }
LABEL_32:
        v49 += 8LL;
      }
      while ( v47 != (_BYTE *)v49 );
      if ( !v46 )
      {
LABEL_34:
        v30 = 0;
        sub_31C00E0((__int64)a1, v48);
        goto LABEL_41;
      }
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)v11;
      if ( v10 - v11 > 8 )
        break;
      a1[2] -= 8;
      v13 = *(_QWORD *)(v12 + 32);
      if ( v13 )
        goto LABEL_5;
LABEL_39:
      v50 = *(_QWORD *)(v12 + 8);
      v34 = sub_31C1720((__int64)a1, &v50, 1u, v7, v8, (__int64)v9);
      sub_31C05A0((__int64)a1, v34, v35, v36, v37, v38);
LABEL_11:
      v10 = (char *)a1[2];
      v11 = (char *)a1[1];
      if ( v11 == v10 )
        goto LABEL_12;
    }
    v33 = *((_QWORD *)v10 - 1);
    *((_QWORD *)v10 - 1) = v12;
    sub_31C0E70((__int64)v11, 0, (v10 - 8 - v11) >> 3, v33);
    a1[2] -= 8;
    v13 = *(_QWORD *)(v12 + 32);
    if ( !v13 )
      goto LABEL_39;
LABEL_5:
    v14 = *(__int64 **)v13;
    v15 = (_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8));
    if ( v15 != sub_31BFDD0(*(_QWORD **)v13, (__int64)v15) )
    {
      v31 = (unsigned int)v52;
      v7 = HIDWORD(v52);
      v32 = (unsigned int)v52 + 1LL;
      if ( v32 > HIDWORD(v52) )
      {
        sub_C8D5F0((__int64)&v51, v53, v32, 8u, v8, (__int64)v9);
        v31 = (unsigned int)v52;
      }
      *(_QWORD *)&v51[8 * v31] = v12;
      LODWORD(v52) = v52 + 1;
      continue;
    }
    break;
  }
  for ( ; v15 != v14; ++v14 )
  {
    if ( v12 != *v14 )
      sub_31C1020(a1, *v14, v16, v17, v8, (__int64)v9);
  }
  sub_31C05A0((__int64)a1, v13, v16, v17, v8, (__int64)v9);
  if ( v48 != v13 )
    goto LABEL_11;
  v30 = 1;
LABEL_41:
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return v30;
}

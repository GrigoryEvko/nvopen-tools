// Function: sub_14AF8E0
// Address: 0x14af8e0
//
__int64 __fastcall sub_14AF8E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // rcx
  signed __int64 v6; // rdx
  _QWORD *v7; // rdx
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // rax
  char v12; // dl
  __int64 v13; // r14
  unsigned __int64 v14; // r15
  __int64 *v15; // r12
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 *v19; // rcx
  __int64 *v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // r8
  __int64 v24; // r14
  __int64 *v25; // rdx
  __int64 *v26; // rsi
  __int64 *v27; // rcx
  _QWORD *v28; // [rsp+8h] [rbp-2D8h]
  _QWORD *v30; // [rsp+40h] [rbp-2A0h] BYREF
  __int64 v31; // [rsp+48h] [rbp-298h]
  _QWORD v32[16]; // [rsp+50h] [rbp-290h] BYREF
  __int64 v33; // [rsp+D0h] [rbp-210h] BYREF
  __int64 *v34; // [rsp+D8h] [rbp-208h]
  __int64 *v35; // [rsp+E0h] [rbp-200h]
  __int64 v36; // [rsp+E8h] [rbp-1F8h]
  int v37; // [rsp+F0h] [rbp-1F0h]
  _BYTE v38[136]; // [rsp+F8h] [rbp-1E8h] BYREF
  __int64 v39; // [rsp+180h] [rbp-160h] BYREF
  __int64 *v40; // [rsp+188h] [rbp-158h]
  __int64 *v41; // [rsp+190h] [rbp-150h]
  __int64 v42; // [rsp+198h] [rbp-148h]
  int v43; // [rsp+1A0h] [rbp-140h]
  _BYTE v44[312]; // [rsp+1A8h] [rbp-138h] BYREF

  v30 = v32;
  v31 = 0x1000000001LL;
  v40 = (__int64 *)v44;
  v41 = (__int64 *)v44;
  v34 = (__int64 *)v38;
  v35 = (__int64 *)v38;
  v2 = *(_DWORD *)(a1 + 20);
  v32[0] = a1;
  v39 = 0;
  v3 = 24LL * (v2 & 0xFFFFFFF);
  v42 = 32;
  v43 = 0;
  v33 = 0;
  v36 = 16;
  v37 = 0;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v4 = *(_QWORD **)(a1 - 8);
    v5 = &v4[(unsigned __int64)v3 / 8];
  }
  else
  {
    v5 = (_QWORD *)a1;
    v4 = (_QWORD *)(a1 - v3);
  }
  v6 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( v6 >> 2 )
  {
    v7 = &v4[12 * (v6 >> 2)];
    while ( 1 )
    {
      if ( a2 == *v4 )
        goto LABEL_10;
      if ( a2 == v4[3] )
      {
        v4 += 3;
        goto LABEL_10;
      }
      if ( a2 == v4[6] )
      {
        v4 += 6;
        goto LABEL_10;
      }
      if ( a2 == v4[9] )
        break;
      v4 += 12;
      if ( v7 == v4 )
      {
        v6 = 0xAAAAAAAAAAAAAAABLL * (v5 - v4);
        goto LABEL_78;
      }
    }
    v4 += 9;
LABEL_10:
    v8 = 1;
    if ( v5 != v4 )
      goto LABEL_38;
    goto LABEL_11;
  }
LABEL_78:
  if ( v6 != 2 )
  {
    if ( v6 != 3 )
    {
      if ( v6 != 1 )
        goto LABEL_11;
      goto LABEL_81;
    }
    if ( a2 == *v4 )
      goto LABEL_10;
    v4 += 3;
  }
  if ( a2 == *v4 )
    goto LABEL_10;
  v4 += 3;
LABEL_81:
  if ( a2 == *v4 )
    goto LABEL_10;
LABEL_11:
  LODWORD(v9) = 1;
  while ( 1 )
  {
    if ( !(_DWORD)v9 )
    {
LABEL_33:
      v8 = 0;
      goto LABEL_34;
    }
    while ( 1 )
    {
      v10 = v30[(unsigned int)v9 - 1];
      LODWORD(v31) = v9 - 1;
      v11 = v40;
      if ( v41 == v40 )
      {
        v19 = &v40[HIDWORD(v42)];
        if ( v40 != v19 )
        {
          v20 = 0;
          do
          {
            while ( 1 )
            {
              if ( v10 == *v11 )
                goto LABEL_32;
              if ( *v11 != -2 )
                break;
              v20 = v11;
              if ( v19 == v11 + 1 )
                goto LABEL_50;
              ++v11;
            }
            ++v11;
          }
          while ( v19 != v11 );
          if ( v20 )
          {
LABEL_50:
            *v20 = v10;
            v14 = (unsigned __int64)v35;
            --v43;
            v13 = *(_QWORD *)(v10 + 8);
            ++v39;
            if ( !v13 )
              goto LABEL_51;
LABEL_22:
            while ( 2 )
            {
              v17 = sub_1648700(v13);
              v16 = v34;
              if ( (__int64 *)v14 == v34 )
              {
                v15 = (__int64 *)(v14 + 8LL * HIDWORD(v36));
                if ( (__int64 *)v14 == v15 )
                {
                  v25 = (__int64 *)v14;
                }
                else
                {
                  do
                  {
                    if ( v17 == *v16 )
                      break;
                    ++v16;
                  }
                  while ( v15 != v16 );
                  v25 = (__int64 *)(v14 + 8LL * HIDWORD(v36));
                }
                goto LABEL_28;
              }
              v15 = (__int64 *)(v14 + 8LL * (unsigned int)v36);
              v16 = (__int64 *)sub_16CC9F0(&v33, v17);
              if ( v17 == *v16 )
              {
                v14 = (unsigned __int64)v35;
                if ( v35 == v34 )
                  v25 = &v35[HIDWORD(v36)];
                else
                  v25 = &v35[(unsigned int)v36];
              }
              else
              {
                v14 = (unsigned __int64)v35;
                if ( v35 != v34 )
                {
                  v16 = &v35[(unsigned int)v36];
                  goto LABEL_20;
                }
                v16 = &v35[HIDWORD(v36)];
                v25 = v16;
              }
LABEL_28:
              if ( v16 != v25 )
              {
                while ( (unsigned __int64)*v16 >= 0xFFFFFFFFFFFFFFFELL )
                {
                  if ( v25 == ++v16 )
                  {
                    if ( v15 != v16 )
                      goto LABEL_21;
                    goto LABEL_32;
                  }
                }
              }
LABEL_20:
              if ( v15 == v16 )
                goto LABEL_32;
LABEL_21:
              v13 = *(_QWORD *)(v13 + 8);
              if ( !v13 )
                goto LABEL_51;
              continue;
            }
          }
        }
        if ( HIDWORD(v42) < (unsigned int)v42 )
          break;
      }
      sub_16CCBA0(&v39, v10);
      if ( v12 )
        goto LABEL_15;
LABEL_32:
      LODWORD(v9) = v31;
      if ( !(_DWORD)v31 )
        goto LABEL_33;
    }
    ++HIDWORD(v42);
    *v19 = v10;
    ++v39;
LABEL_15:
    v13 = *(_QWORD *)(v10 + 8);
    v14 = (unsigned __int64)v35;
    if ( v13 )
      goto LABEL_22;
LABEL_51:
    if ( a2 == v10 )
      break;
    if ( a1 != v10 && !(unsigned __int8)sub_14AF470(v10, 0, 0, 0) )
      goto LABEL_32;
    v21 = v34;
    if ( v35 != v34 )
    {
LABEL_55:
      sub_16CCBA0(&v33, v10);
      goto LABEL_56;
    }
    v26 = &v34[HIDWORD(v36)];
    if ( v34 == v26 )
    {
LABEL_91:
      if ( HIDWORD(v36) < (unsigned int)v36 )
      {
        ++HIDWORD(v36);
        *v26 = v10;
        ++v33;
        goto LABEL_56;
      }
      goto LABEL_55;
    }
    v27 = 0;
    while ( v10 != *v21 )
    {
      if ( *v21 == -2 )
        v27 = v21;
      if ( v26 == ++v21 )
      {
        if ( !v27 )
          goto LABEL_91;
        *v27 = v10;
        --v37;
        ++v33;
        break;
      }
    }
LABEL_56:
    v9 = (unsigned int)v31;
    if ( (unsigned __int8)(*(_BYTE *)(v10 + 16) - 17) > 6u )
    {
      v22 = 3LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      v23 = (_QWORD *)(v10 - v22 * 8);
      if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      {
        v23 = *(_QWORD **)(v10 - 8);
        v10 = (__int64)&v23[v22];
      }
      if ( (_QWORD *)v10 != v23 )
      {
        do
        {
          v24 = *v23;
          if ( HIDWORD(v31) <= (unsigned int)v9 )
          {
            v28 = v23;
            sub_16CD150(&v30, v32, 0, 8);
            v9 = (unsigned int)v31;
            v23 = v28;
          }
          v23 += 3;
          v30[v9] = v24;
          v9 = (unsigned int)(v31 + 1);
          LODWORD(v31) = v31 + 1;
        }
        while ( v23 != (_QWORD *)v10 );
      }
    }
  }
  v8 = 1;
LABEL_34:
  if ( v35 != v34 )
    _libc_free((unsigned __int64)v35);
  if ( v41 != v40 )
    _libc_free((unsigned __int64)v41);
LABEL_38:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v8;
}

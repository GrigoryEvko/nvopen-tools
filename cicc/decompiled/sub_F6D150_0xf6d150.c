// Function: sub_F6D150
// Address: 0xf6d150
//
__int64 __fastcall sub_F6D150(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, char a5)
{
  __int64 **v5; // rsi
  __int64 **v6; // rax
  __int64 v7; // r14
  unsigned __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 *v17; // rdx
  __int64 *v19; // rax
  char v20; // dl
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r12
  char v25; // r14
  __int64 *v26; // r13
  __int64 **v27; // rax
  __int64 **v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 *v33; // rax
  char v34; // r13
  __int64 v35; // r14
  __int64 **v39; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v41; // [rsp+2Bh] [rbp-D5h]
  __int64 **v42; // [rsp+40h] [rbp-C0h]
  __int64 v43; // [rsp+48h] [rbp-B8h]
  __int64 v44; // [rsp+50h] [rbp-B0h]
  int v45; // [rsp+58h] [rbp-A8h]
  unsigned int v46; // [rsp+5Ch] [rbp-A4h]
  unsigned int v47; // [rsp+5Ch] [rbp-A4h]
  __int64 **v48; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+68h] [rbp-98h]
  _BYTE v50[32]; // [rsp+70h] [rbp-90h] BYREF
  __int64 v51; // [rsp+90h] [rbp-70h] BYREF
  __int64 *v52; // [rsp+98h] [rbp-68h]
  __int64 v53; // [rsp+A0h] [rbp-60h]
  int v54; // [rsp+A8h] [rbp-58h]
  char v55; // [rsp+ACh] [rbp-54h]
  char v56; // [rsp+B0h] [rbp-50h] BYREF

  v48 = (__int64 **)v50;
  v49 = 0x400000000LL;
  v5 = *(__int64 ***)(a1 + 40);
  v52 = (__int64 *)&v56;
  v6 = *(__int64 ***)(a1 + 32);
  v51 = 0;
  v53 = 4;
  v54 = 0;
  v55 = 1;
  v39 = v5;
  if ( v5 == v6 )
    return 0;
  v42 = v6;
  v7 = a1;
  v44 = a1 + 56;
  v41 = 0;
  do
  {
    v8 = (*v42)[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v8 != *v42 + 6 )
    {
      if ( !v8 )
        BUG();
      v43 = v8 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
      {
        v45 = sub_B46E30(v8 - 24);
        if ( v45 )
        {
          v9 = 0;
          v10 = v7;
          while ( 1 )
          {
            v5 = (__int64 **)v9;
            v11 = sub_B46EC0(v43, v9);
            v15 = v11;
            if ( *(_BYTE *)(v10 + 84) )
            {
              v16 = *(__int64 **)(v10 + 64);
              v17 = &v16[*(unsigned int *)(v10 + 76)];
              if ( v16 != v17 )
              {
                while ( v15 != *v16 )
                {
                  if ( v17 == ++v16 )
                    goto LABEL_22;
                }
                goto LABEL_13;
              }
            }
            else
            {
              v5 = (__int64 **)v11;
              if ( sub_C8CA60(v44, v11) )
                goto LABEL_13;
            }
LABEL_22:
            if ( !v55 )
              goto LABEL_28;
            v19 = v52;
            v12 = HIDWORD(v53);
            v17 = &v52[HIDWORD(v53)];
            if ( v52 != v17 )
            {
              while ( v15 != *v19 )
              {
                if ( v17 == ++v19 )
                  goto LABEL_55;
              }
              goto LABEL_13;
            }
LABEL_55:
            if ( HIDWORD(v53) < (unsigned int)v53 )
            {
              ++HIDWORD(v53);
              *v17 = v15;
              ++v51;
LABEL_29:
              v21 = *(_QWORD *)(v15 + 16);
              if ( v21 )
              {
                while ( 1 )
                {
                  v22 = *(_QWORD *)(v21 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
                    break;
                  v21 = *(_QWORD *)(v21 + 8);
                  if ( !v21 )
                  {
                    LODWORD(v49) = 0;
                    goto LABEL_51;
                  }
                }
                v23 = v9;
                v24 = v15;
                v25 = 1;
                v26 = *(__int64 **)(v22 + 40);
                if ( !*(_BYTE *)(v10 + 84) )
                  goto LABEL_46;
                while ( 1 )
                {
                  v27 = *(__int64 ***)(v10 + 64);
                  v28 = &v27[*(unsigned int *)(v10 + 76)];
                  if ( v27 == v28 )
                    break;
                  while ( v26 != *v27 )
                  {
                    if ( v28 == ++v27 )
                      goto LABEL_47;
                  }
                  while ( 1 )
                  {
                    v29 = v26[6] & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (__int64 *)v29 == v26 + 6 || !v29 || (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
                      BUG();
                    if ( *(_BYTE *)(v29 - 24) == 33 )
                    {
                      LODWORD(v49) = 0;
                      v9 = v23;
                      goto LABEL_51;
                    }
                    v30 = (unsigned int)v49;
                    v31 = (unsigned int)v49 + 1LL;
                    if ( v31 > HIDWORD(v49) )
                    {
                      v5 = (__int64 **)v50;
                      v47 = v23;
                      sub_C8D5F0((__int64)&v48, v50, v31, 8u, v23, v14);
                      v30 = (unsigned int)v49;
                      v23 = v47;
                    }
                    v48[v30] = v26;
                    LODWORD(v49) = v49 + 1;
                    do
                    {
                      v21 = *(_QWORD *)(v21 + 8);
                      if ( !v21 )
                        goto LABEL_48;
LABEL_44:
                      v32 = *(_QWORD *)(v21 + 24);
                    }
                    while ( (unsigned __int8)(*(_BYTE *)v32 - 30) > 0xAu );
                    v26 = *(__int64 **)(v32 + 40);
                    if ( *(_BYTE *)(v10 + 84) )
                      break;
LABEL_46:
                    v5 = (__int64 **)v26;
                    v46 = v23;
                    v33 = sub_C8CA60(v44, (__int64)v26);
                    v23 = v46;
                    if ( !v33 )
                      goto LABEL_47;
                  }
                }
LABEL_47:
                v21 = *(_QWORD *)(v21 + 8);
                v25 = 0;
                if ( v21 )
                  goto LABEL_44;
LABEL_48:
                v34 = v25;
                v35 = v24;
                v9 = v23;
                if ( !v34 )
                {
                  v5 = v48;
                  sub_F40FB0(v35, v48, (unsigned int)v49, ".loopexit", a2, a3, a4, a5);
                  v41 = 1;
                }
              }
              LODWORD(v49) = 0;
LABEL_51:
              if ( v45 == ++v9 )
              {
LABEL_14:
                v7 = v10;
                break;
              }
            }
            else
            {
LABEL_28:
              v5 = (__int64 **)v15;
              sub_C8CC70((__int64)&v51, v15, (__int64)v17, v12, v13, v14);
              if ( v20 )
                goto LABEL_29;
LABEL_13:
              if ( v45 == ++v9 )
                goto LABEL_14;
            }
          }
        }
      }
    }
    ++v42;
  }
  while ( v39 != v42 );
  if ( !v55 )
    _libc_free(v52, v5);
  if ( v48 != (__int64 **)v50 )
    _libc_free(v48, v5);
  return v41;
}

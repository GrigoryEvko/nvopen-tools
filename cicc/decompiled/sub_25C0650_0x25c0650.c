// Function: sub_25C0650
// Address: 0x25c0650
//
__int64 __fastcall sub_25C0650(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  char v8; // r12
  __int64 v9; // r14
  __int64 *v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rcx
  unsigned __int8 *v15; // r15
  unsigned __int8 *v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r15
  char v19; // r14
  __int64 *v20; // rax
  __int64 v21; // rsi
  __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r12
  __int64 v32; // r8
  __int64 v33; // r9
  char v34; // dl
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  char v37; // dl
  __int64 v38; // rcx
  unsigned __int64 v39; // r8
  __int64 v40; // r9
  __int64 *v41; // r12
  __int64 v42; // rdx
  __int64 v43; // [rsp+10h] [rbp-2C0h]
  char v44; // [rsp+10h] [rbp-2C0h]
  char v45; // [rsp+10h] [rbp-2C0h]
  char v46; // [rsp+18h] [rbp-2B8h]
  __int64 v47; // [rsp+28h] [rbp-2A8h]
  char v49; // [rsp+3Fh] [rbp-291h]
  _BYTE v50[48]; // [rsp+40h] [rbp-290h] BYREF
  __int64 *v51; // [rsp+70h] [rbp-260h] BYREF
  __int64 v52; // [rsp+78h] [rbp-258h]
  _BYTE v53[256]; // [rsp+80h] [rbp-250h] BYREF
  __int64 v54; // [rsp+180h] [rbp-150h] BYREF
  __int64 *v55; // [rsp+188h] [rbp-148h]
  __int64 v56; // [rsp+190h] [rbp-140h]
  int v57; // [rsp+198h] [rbp-138h]
  char v58; // [rsp+19Ch] [rbp-134h]
  char v59; // [rsp+1A0h] [rbp-130h] BYREF

  v52 = 0x2000000000LL;
  v51 = (__int64 *)v53;
  v54 = 0;
  v55 = (__int64 *)&v59;
  v56 = 32;
  v57 = 0;
  v58 = 1;
  if ( (unsigned __int8)sub_B2D6A0(a1) )
    goto LABEL_2;
  v8 = sub_B2D6C0(a1);
  if ( v8 )
    goto LABEL_2;
  v9 = *(_QWORD *)(a1 + 16);
  if ( v9 )
  {
    while ( 1 )
    {
      if ( !v58 )
        goto LABEL_48;
      v10 = v55;
      v5 = HIDWORD(v56);
      v4 = &v55[HIDWORD(v56)];
      if ( v55 != v4 )
      {
        while ( *v10 != v9 )
        {
          if ( v4 == ++v10 )
            goto LABEL_54;
        }
        goto LABEL_15;
      }
LABEL_54:
      if ( HIDWORD(v56) < (unsigned int)v56 )
      {
        ++HIDWORD(v56);
        *v4 = v9;
        ++v54;
      }
      else
      {
LABEL_48:
        sub_C8CC70((__int64)&v54, v9, (__int64)v4, v5, v6, v7);
      }
LABEL_15:
      v11 = (unsigned int)v52;
      v5 = HIDWORD(v52);
      v12 = (unsigned int)v52 + 1LL;
      if ( v12 > HIDWORD(v52) )
      {
        sub_C8D5F0((__int64)&v51, v53, v12, 8u, v6, v7);
        v11 = (unsigned int)v52;
      }
      v4 = v51;
      v51[v11] = v9;
      v13 = v52 + 1;
      LODWORD(v52) = v52 + 1;
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_18;
    }
  }
  v13 = v52;
LABEL_18:
  if ( !v13 )
  {
    v2 = 50;
    goto LABEL_3;
  }
  v49 = 0;
  while ( 2 )
  {
    v14 = v13--;
    v15 = (unsigned __int8 *)v51[v14 - 1];
    LODWORD(v52) = v13;
    v16 = (unsigned __int8 *)*((_QWORD *)v15 + 3);
    v17 = (unsigned int)*v16 - 29;
    switch ( *v16 )
    {
      case 0x1Eu:
      case 0x52u:
        v19 = v8 & v49;
        goto LABEL_29;
      case 0x22u:
      case 0x55u:
        if ( v15 == v16 - 32 )
          goto LABEL_43;
        v21 = 0;
        v47 = (v15 - &v16[-32 * (*((_DWORD *)v16 + 1) & 0x7FFFFFF)]) >> 5;
        if ( sub_98AB90(*((_QWORD *)v15 + 3), 0) )
        {
          if ( *((_QWORD *)v16 + 2) )
          {
            v44 = v8;
            v31 = *((_QWORD *)v16 + 2);
            do
            {
              v21 = v31;
              sub_AE6EC0((__int64)&v54, v31);
              if ( v34 )
              {
                v35 = (unsigned int)v52;
                v36 = (unsigned int)v52 + 1LL;
                if ( v36 > HIDWORD(v52) )
                {
                  v21 = (__int64)v53;
                  sub_C8D5F0((__int64)&v51, v53, v36, 8u, v32, v33);
                  v35 = (unsigned int)v52;
                }
                v51[v35] = v31;
                LODWORD(v52) = v52 + 1;
              }
              v31 = *(_QWORD *)(v31 + 8);
            }
            while ( v31 );
            v8 = v44;
          }
        }
        else
        {
          v22 = sub_B49EE0(v16, v47);
          v21 = HIBYTE(v22);
          LOBYTE(v21) = v22 | HIBYTE(v22);
          if ( v22 )
          {
            if ( !(unsigned __int8)sub_B49E20((__int64)v16) )
              goto LABEL_2;
            if ( *(_BYTE *)(*((_QWORD *)v16 + 1) + 8LL) != 7 && *((_QWORD *)v16 + 2) )
            {
              v45 = v8;
              v41 = (__int64 *)*((_QWORD *)v16 + 2);
              do
              {
                v21 = (__int64)&v54;
                sub_25C0560((__int64)v50, (__int64)&v54, v41, v38, v39, v40);
                if ( v50[32] )
                {
                  v42 = (unsigned int)v52;
                  v39 = (unsigned int)v52 + 1LL;
                  if ( v39 > HIDWORD(v52) )
                  {
                    v21 = (__int64)v53;
                    sub_C8D5F0((__int64)&v51, v53, (unsigned int)v52 + 1LL, 8u, v39, v40);
                    v42 = (unsigned int)v52;
                  }
                  v38 = (__int64)v51;
                  v51[v42] = (__int64)v41;
                  LODWORD(v52) = v52 + 1;
                }
                v41 = (__int64 *)v41[1];
              }
              while ( v41 );
              v8 = v45;
            }
          }
        }
        v46 = sub_B49D00((__int64)v16);
        if ( (v46 & 3) == 0 )
          goto LABEL_68;
        v23 = *((_QWORD *)v16 - 4);
        if ( !v23 )
          goto LABEL_39;
        if ( *(_BYTE *)v23 )
          goto LABEL_39;
        if ( *(_QWORD *)(v23 + 24) != *((_QWORD *)v16 + 10) )
          goto LABEL_39;
        if ( v15 < &v16[-32 * (*((_DWORD *)v16 + 1) & 0x7FFFFFF)] )
          goto LABEL_39;
        v43 = *((_QWORD *)v16 - 4);
        if ( v15 >= sub_24E54B0(v16) )
          goto LABEL_39;
        v27 = v43;
        if ( (unsigned __int64)(unsigned int)v47 >= *(_QWORD *)(v43 + 104) )
          goto LABEL_39;
        if ( (*(_BYTE *)(v43 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v43, v21, v43, v26);
          v27 = v43;
        }
        v28 = *(_QWORD *)(v27 + 96) + 40LL * (unsigned int)v47;
        if ( *(_BYTE *)(a2 + 28) )
        {
          v29 = *(_QWORD **)(a2 + 8);
          v30 = &v29[*(unsigned int *)(a2 + 20)];
          if ( v29 != v30 )
          {
            while ( v28 != *v29 )
            {
              if ( v30 == ++v29 )
                goto LABEL_39;
            }
LABEL_68:
            v19 = v8 & v49;
LABEL_28:
            v13 = v52;
            goto LABEL_29;
          }
        }
        else if ( sub_C8CA60(a2, v28) )
        {
          goto LABEL_68;
        }
LABEL_39:
        if ( sub_CF49B0(v16, v47, 50) )
          goto LABEL_68;
        if ( (v46 & 2) != 0 )
        {
          if ( (unsigned int)v47 < (unsigned int)((sub_24E54B0(v16) - &v16[-32 * (*((_DWORD *)v16 + 1) & 0x7FFFFFF)]) >> 5)
            && (v37 = sub_B49B80((__int64)v16, v47, 81)) != 0
            || (v37 = sub_CF49B0(v16, v47, 51)) != 0
            || (v37 = sub_CF49B0(v16, v47, 50)) != 0 )
          {
            v49 = v37;
            v13 = v52;
            v19 = v8;
          }
          else if ( (v46 & 1) != 0 )
          {
            v8 = sub_CF49B0(v16, v47, 78);
            if ( !v8 )
              goto LABEL_2;
            v13 = v52;
            v19 = v49;
          }
          else
          {
            v13 = v52;
            v19 = v49;
            v8 = 1;
          }
        }
        else
        {
          v49 = 1;
          v13 = v52;
          v19 = v8;
        }
LABEL_29:
        if ( v13 )
        {
          if ( v19 )
            goto LABEL_2;
          continue;
        }
        if ( v19 )
        {
LABEL_2:
          v2 = 0;
          goto LABEL_3;
        }
        v2 = 51;
        if ( !v49 )
          v2 = v8 == 0 ? 50 : 78;
LABEL_3:
        if ( !v58 )
          _libc_free((unsigned __int64)v55);
        if ( v51 != (__int64 *)v53 )
          _libc_free((unsigned __int64)v51);
        return v2;
      case 0x3Du:
        if ( (v16[2] & 1) != 0 )
          goto LABEL_2;
LABEL_43:
        v49 = 1;
        v19 = v8;
        goto LABEL_29;
      case 0x3Eu:
        if ( *(_QWORD *)v15 == *((_QWORD *)v16 - 8) || (v16[2] & 1) != 0 )
          goto LABEL_2;
        v19 = v49;
        v8 = 1;
        goto LABEL_29;
      case 0x3Fu:
      case 0x4Eu:
      case 0x4Fu:
      case 0x54u:
      case 0x56u:
        v18 = *((_QWORD *)v16 + 2);
        v19 = v8 & v49;
        if ( !v18 )
          goto LABEL_29;
        do
        {
          while ( 2 )
          {
            if ( !v58 )
              goto LABEL_49;
            v20 = v55;
            v14 = HIDWORD(v56);
            v17 = (__int64)&v55[HIDWORD(v56)];
            if ( v55 != (__int64 *)v17 )
            {
              while ( v18 != *v20 )
              {
                if ( (__int64 *)v17 == ++v20 )
                  goto LABEL_56;
              }
LABEL_27:
              v18 = *(_QWORD *)(v18 + 8);
              if ( !v18 )
                goto LABEL_28;
              continue;
            }
            break;
          }
LABEL_56:
          if ( HIDWORD(v56) < (unsigned int)v56 )
          {
            ++HIDWORD(v56);
            *(_QWORD *)v17 = v18;
            ++v54;
          }
          else
          {
LABEL_49:
            sub_C8CC70((__int64)&v54, v18, v17, v14, v6, v7);
            if ( !(_BYTE)v17 )
              goto LABEL_27;
          }
          v24 = (unsigned int)v52;
          v14 = HIDWORD(v52);
          v25 = (unsigned int)v52 + 1LL;
          if ( v25 > HIDWORD(v52) )
          {
            sub_C8D5F0((__int64)&v51, v53, v25, 8u, v6, v7);
            v24 = (unsigned int)v52;
          }
          v17 = (__int64)v51;
          v51[v24] = v18;
          LODWORD(v52) = v52 + 1;
          v18 = *(_QWORD *)(v18 + 8);
        }
        while ( v18 );
        goto LABEL_28;
      default:
        goto LABEL_2;
    }
  }
}

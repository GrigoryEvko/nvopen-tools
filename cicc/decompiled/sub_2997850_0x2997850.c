// Function: sub_2997850
// Address: 0x2997850
//
void __fastcall sub_2997850(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 v16; // r14
  unsigned __int8 *v17; // r15
  unsigned __int8 **v18; // rax
  __int64 v19; // r14
  __int64 *v20; // rax
  unsigned __int64 v21; // rdx
  bool v22; // r14
  unsigned __int8 **v23; // rax
  unsigned __int8 **v24; // rax
  unsigned __int8 **v25; // rdx
  __int64 v26; // rcx
  unsigned __int8 **v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  int v35; // eax
  __int64 v36; // rax
  __int16 v37; // ax
  int v38; // [rsp+0h] [rbp-280h]
  __int64 v39; // [rsp+8h] [rbp-278h]
  __int64 *v40; // [rsp+20h] [rbp-260h] BYREF
  __int64 v41; // [rsp+28h] [rbp-258h]
  _BYTE v42[256]; // [rsp+30h] [rbp-250h] BYREF
  __int64 v43; // [rsp+130h] [rbp-150h] BYREF
  __int64 *v44; // [rsp+138h] [rbp-148h]
  __int64 v45; // [rsp+140h] [rbp-140h]
  int v46; // [rsp+148h] [rbp-138h]
  char v47; // [rsp+14Ch] [rbp-134h]
  char v48; // [rsp+150h] [rbp-130h] BYREF

  v7 = *(_QWORD *)(a2 + 16);
  v40 = (__int64 *)v42;
  v41 = 0x2000000000LL;
  v44 = (__int64 *)&v48;
  v43 = 0;
  v45 = 32;
  v46 = 0;
  v47 = 1;
  if ( v7 )
  {
LABEL_2:
    v8 = v44;
    v9 = HIDWORD(v45);
    v10 = &v44[HIDWORD(v45)];
    if ( v44 == v10 )
    {
LABEL_15:
      if ( HIDWORD(v45) >= (unsigned int)v45 )
        goto LABEL_8;
      ++HIDWORD(v45);
      *v10 = v7;
      ++v43;
LABEL_9:
      v11 = (unsigned int)v41;
      v9 = HIDWORD(v41);
      v12 = (unsigned int)v41 + 1LL;
      if ( v12 > HIDWORD(v41) )
      {
        sub_C8D5F0((__int64)&v40, v42, v12, 8u, a5, a6);
        v11 = (unsigned int)v41;
      }
      v10 = v40;
      v40[v11] = v7;
      LODWORD(v41) = v41 + 1;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v7 )
        goto LABEL_7;
    }
    else
    {
      while ( *v8 != v7 )
      {
        if ( v10 == ++v8 )
          goto LABEL_15;
      }
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
LABEL_7:
        if ( v47 )
          goto LABEL_2;
LABEL_8:
        sub_C8CC70((__int64)&v43, v7, (__int64)v10, v9, a5, a6);
        if ( (_BYTE)v10 )
          goto LABEL_9;
      }
    }
    v13 = v40;
    LODWORD(v14) = v41;
    goto LABEL_13;
  }
  v13 = (__int64 *)v42;
  LODWORD(v14) = 0;
LABEL_13:
  while ( 1 )
  {
    v15 = (__int64)&v13[(unsigned int)v14];
    if ( !(_DWORD)v14 )
      break;
    while ( 2 )
    {
      v16 = *(_QWORD *)(v15 - 8);
      v14 = (unsigned int)(v14 - 1);
      LODWORD(v41) = v14;
      v17 = *(unsigned __int8 **)(v16 + 24);
      switch ( *v17 )
      {
        case '"':
        case 'U':
          v21 = (unsigned __int64)&v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
          if ( v16 < v21 )
            goto LABEL_31;
          if ( *v17 == 40 )
          {
            v39 = -32 - 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(v16 + 24));
            if ( (v17[7] & 0x80u) == 0 )
              goto LABEL_76;
          }
          else
          {
            v39 = -32;
            if ( *v17 == 85 )
            {
              if ( (v17[7] & 0x80u) == 0 )
                goto LABEL_76;
            }
            else
            {
              if ( *v17 != 34 )
LABEL_105:
                BUG();
              v39 = -96;
              if ( (v17[7] & 0x80u) == 0 )
                goto LABEL_76;
            }
          }
          v30 = sub_BD2BC0((__int64)v17);
          if ( (v17[7] & 0x80u) == 0 )
          {
            if ( (unsigned int)((v30 + v31) >> 4) )
LABEL_103:
              BUG();
          }
          else if ( (unsigned int)((v30 + v31 - sub_BD2BC0((__int64)v17)) >> 4) )
          {
            if ( (v17[7] & 0x80u) == 0 )
              goto LABEL_103;
            v38 = *(_DWORD *)(sub_BD2BC0((__int64)v17) + 8);
            if ( (v17[7] & 0x80u) == 0 )
              BUG();
            v32 = sub_BD2BC0((__int64)v17);
            v39 -= 32LL * (unsigned int)(*(_DWORD *)(v32 + v33 - 4) - v38);
          }
LABEL_76:
          v34 = *((_DWORD *)v17 + 1);
          v15 = (__int64)&v17[-32 * (v34 & 0x7FFFFFF)];
          if ( v16 >= (unsigned __int64)&v17[v39] )
            goto LABEL_79;
          if ( (unsigned __int8)sub_B49B80((__int64)v17, (__int64)(v16 - v15) >> 5, 81) )
            goto LABEL_29;
          v34 = *((_DWORD *)v17 + 1);
LABEL_79:
          v21 = (unsigned __int64)&v17[-32 * (v34 & 0x7FFFFFF)];
          if ( v16 < v21 )
            goto LABEL_31;
          v35 = *v17;
          v21 = (unsigned int)(v35 - 29);
          if ( v35 == 40 )
          {
            v21 = 32LL * (unsigned int)sub_B491D0((__int64)v17);
            v36 = -32LL - v21;
          }
          else
          {
            v36 = -32;
            if ( (_DWORD)v21 != 56 )
            {
              if ( (_DWORD)v21 != 5 )
                goto LABEL_105;
              v36 = -96;
            }
          }
          if ( v16 >= (unsigned __int64)&v17[v36] )
          {
LABEL_31:
            v22 = 0;
          }
          else
          {
            v37 = sub_B49EE0(v17, (__int64)(v16 - (_QWORD)&v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)]) >> 5);
            v21 = HIBYTE(v37);
            v22 = v37 == 0;
          }
          if ( !*(_BYTE *)(a1 + 28) )
            goto LABEL_67;
          v23 = *(unsigned __int8 ***)(a1 + 8);
          v15 = *(unsigned int *)(a1 + 20);
          v21 = (unsigned __int64)&v23[v15];
          if ( v23 == (unsigned __int8 **)v21 )
          {
LABEL_90:
            if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 16) )
            {
LABEL_67:
              sub_C8CC70(a1, (__int64)v17, v21, v15, a5, a6);
            }
            else
            {
              *(_DWORD *)(a1 + 20) = v15 + 1;
              *(_QWORD *)v21 = v17;
              ++*(_QWORD *)a1;
            }
          }
          else
          {
            while ( v17 != *v23 )
            {
              if ( (unsigned __int8 **)v21 == ++v23 )
                goto LABEL_90;
            }
          }
          if ( v22 )
            goto LABEL_29;
          if ( (unsigned __int8)sub_B49E20((__int64)v17) )
            goto LABEL_22;
          if ( !*(_BYTE *)(a1 + 316) )
            goto LABEL_66;
          v24 = *(unsigned __int8 ***)(a1 + 296);
          v15 = *(unsigned int *)(a1 + 308);
          v14 = (__int64)&v24[v15];
          if ( v24 == (unsigned __int8 **)v14 )
          {
LABEL_65:
            if ( (unsigned int)v15 < *(_DWORD *)(a1 + 304) )
            {
              v15 = (unsigned int)(v15 + 1);
              *(_DWORD *)(a1 + 308) = v15;
              *(_QWORD *)v14 = v17;
              ++*(_QWORD *)(a1 + 288);
            }
            else
            {
LABEL_66:
              sub_C8CC70(a1 + 288, (__int64)v17, v14, v15, a5, a6);
            }
          }
          else
          {
            while ( v17 != *v24 )
            {
              if ( (unsigned __int8 **)v14 == ++v24 )
                goto LABEL_65;
            }
          }
LABEL_22:
          v19 = *((_QWORD *)v17 + 2);
          if ( v19 )
          {
            while ( 1 )
            {
              if ( v47 )
              {
                v20 = v44;
                v15 = HIDWORD(v45);
                v14 = (__int64)&v44[HIDWORD(v45)];
                if ( v44 != (__int64 *)v14 )
                {
                  while ( v19 != *v20 )
                  {
                    if ( (__int64 *)v14 == ++v20 )
                      goto LABEL_63;
                  }
                  goto LABEL_28;
                }
LABEL_63:
                if ( HIDWORD(v45) < (unsigned int)v45 )
                {
                  ++HIDWORD(v45);
                  *(_QWORD *)v14 = v19;
                  ++v43;
                  goto LABEL_59;
                }
              }
              sub_C8CC70((__int64)&v43, v19, v14, v15, a5, a6);
              if ( (_BYTE)v14 )
              {
LABEL_59:
                v28 = (unsigned int)v41;
                v15 = HIDWORD(v41);
                v29 = (unsigned int)v41 + 1LL;
                if ( v29 > HIDWORD(v41) )
                {
                  sub_C8D5F0((__int64)&v40, v42, v29, 8u, a5, a6);
                  v28 = (unsigned int)v41;
                }
                v14 = (__int64)v40;
                v40[v28] = v19;
                LODWORD(v41) = v41 + 1;
                v19 = *(_QWORD *)(v19 + 8);
                if ( !v19 )
                  break;
              }
              else
              {
LABEL_28:
                v19 = *(_QWORD *)(v19 + 8);
                if ( !v19 )
                  break;
              }
            }
          }
LABEL_29:
          v13 = v40;
          LODWORD(v14) = v41;
          break;
        case '=':
          v15 -= 8;
          if ( !(_DWORD)v14 )
            goto LABEL_46;
          continue;
        case '>':
          if ( (unsigned int)sub_BD2910(v16) )
            goto LABEL_29;
          if ( !*(_BYTE *)(a1 + 316) )
            goto LABEL_89;
          v27 = *(unsigned __int8 ***)(a1 + 296);
          v26 = *(unsigned int *)(a1 + 308);
          v25 = &v27[v26];
          if ( v27 == v25 )
            goto LABEL_98;
          while ( v17 != *v27 )
          {
            if ( v25 == ++v27 )
            {
LABEL_98:
              if ( (unsigned int)v26 >= *(_DWORD *)(a1 + 304) )
              {
LABEL_89:
                sub_C8CC70(a1 + 288, (__int64)v17, (__int64)v25, v26, a5, a6);
                v13 = v40;
                LODWORD(v14) = v41;
                goto LABEL_13;
              }
              *(_DWORD *)(a1 + 308) = v26 + 1;
              *v25 = v17;
              ++*(_QWORD *)(a1 + 288);
              goto LABEL_29;
            }
          }
          goto LABEL_29;
        case '?':
        case 'N':
        case 'O':
        case 'T':
        case 'V':
          goto LABEL_22;
        default:
          if ( !*(_BYTE *)(a1 + 316) )
            goto LABEL_66;
          v18 = *(unsigned __int8 ***)(a1 + 296);
          v15 = *(unsigned int *)(a1 + 308);
          v14 = (__int64)&v18[v15];
          if ( v18 == (unsigned __int8 **)v14 )
            goto LABEL_65;
          while ( v17 != *v18 )
          {
            if ( (unsigned __int8 **)v14 == ++v18 )
              goto LABEL_65;
          }
          goto LABEL_22;
      }
      break;
    }
  }
LABEL_46:
  if ( !v47 )
  {
    _libc_free((unsigned __int64)v44);
    v13 = v40;
  }
  if ( v13 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v13);
}

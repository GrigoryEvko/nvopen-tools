// Function: sub_2B2A1B0
// Address: 0x2b2a1b0
//
__int64 __fastcall sub_2B2A1B0(__int64 **a1)
{
  __int64 v2; // rdi
  __int64 v3; // r9
  __int64 v4; // rcx
  __int64 v5; // r8
  int v6; // r10d
  unsigned int v7; // edx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r13
  __int64 result; // rax
  __int64 v14; // rdx
  char v15; // di
  __int64 v16; // rdx
  __int64 *v17; // r14
  char v18; // al
  __int64 *v19; // rbx
  __int64 v20; // rsi
  char *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // rbx
  __int64 v28; // r8
  unsigned __int64 v29; // rsi
  char *v30; // rax
  char *v31; // rcx
  __int64 v32; // r8
  __int64 *v33; // r15
  __int64 v34; // r8
  __int64 v35; // r8
  __int64 v36; // r8
  char *v37; // rsi
  __int64 v38; // rdx
  char *v39; // rax
  int v40; // esi
  __int64 v41; // r8
  char *v42; // rdi
  char *v43; // rcx
  int v44; // r11d
  __int64 *v45; // r8
  bool v46; // zf
  __int64 v47; // [rsp+0h] [rbp-70h] BYREF
  char *v48; // [rsp+8h] [rbp-68h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  int v50; // [rsp+18h] [rbp-58h]
  char v51; // [rsp+1Ch] [rbp-54h]
  char v52; // [rsp+20h] [rbp-50h] BYREF

  v2 = (*a1)[3];
  v3 = *a1[1];
  v4 = *(_BYTE *)(v2 + 88) & 1;
  if ( (*(_BYTE *)(v2 + 88) & 1) != 0 )
  {
    v5 = v2 + 96;
    v6 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(v2 + 104);
    v5 = *(_QWORD *)(v2 + 96);
    if ( !(_DWORD)v14 )
      goto LABEL_28;
    v6 = v14 - 1;
  }
  v7 = v6 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = v5 + 72LL * v7;
  v9 = *(_QWORD *)v8;
  if ( v3 == *(_QWORD *)v8 )
    goto LABEL_4;
  v40 = 1;
  while ( v9 != -4096 )
  {
    v44 = v40 + 1;
    v7 = v6 & (v40 + v7);
    v8 = v5 + 72LL * v7;
    v9 = *(_QWORD *)v8;
    if ( v3 == *(_QWORD *)v8 )
      goto LABEL_4;
    v40 = v44;
  }
  if ( (_BYTE)v4 )
  {
    v22 = 288;
    goto LABEL_29;
  }
  v14 = *(unsigned int *)(v2 + 104);
LABEL_28:
  v22 = 72 * v14;
LABEL_29:
  v8 = v5 + v22;
LABEL_4:
  v10 = 288;
  if ( !(_BYTE)v4 )
    v10 = 72LL * *(unsigned int *)(v2 + 104);
  if ( v8 == v5 + v10 )
    return 0;
  v11 = *(unsigned int *)(v8 + 16);
  v12 = *(__int64 **)(v8 + 8);
  result = 0;
  if ( *(_DWORD *)(v8 + 16) )
  {
    v51 = 1;
    v19 = &v12[v11];
    v15 = 1;
    v47 = 0;
    v48 = &v52;
    v49 = 4;
    v50 = 0;
    while ( 1 )
    {
      v20 = *v12;
      if ( !v15 )
        goto LABEL_11;
      v21 = v48;
      v4 = HIDWORD(v49);
      v11 = (__int64)&v48[8 * HIDWORD(v49)];
      if ( v48 != (char *)v11 )
      {
        while ( v20 != *(_QWORD *)v21 )
        {
          v21 += 8;
          if ( (char *)v11 == v21 )
            goto LABEL_23;
        }
        goto LABEL_12;
      }
LABEL_23:
      if ( HIDWORD(v49) < (unsigned int)v49 )
      {
        v4 = (unsigned int)(HIDWORD(v49) + 1);
        ++v12;
        ++HIDWORD(v49);
        *(_QWORD *)v11 = v20;
        v15 = v51;
        ++v47;
        if ( v19 == v12 )
        {
LABEL_13:
          v17 = (__int64 *)sub_2B2A0E0((*a1)[3], *a1[2]);
          v18 = v51;
          if ( !v16 )
            goto LABEL_14;
          v23 = 8 * v16;
          v24 = &v17[v16];
          v25 = (8 * v16) >> 5;
          v26 = v23 >> 3;
          if ( v25 <= 0 )
            goto LABEL_66;
          v27 = &v17[4 * v25];
          while ( 1 )
          {
            v28 = *v17;
            if ( v51 )
              break;
            if ( sub_C8CA60((__int64)&v47, v28) )
              goto LABEL_99;
            v32 = v17[1];
            v33 = v17 + 1;
            if ( v51 )
            {
              v29 = (unsigned __int64)v48;
              v25 = HIDWORD(v49);
              v31 = v48;
              v30 = &v48[8 * HIDWORD(v49)];
LABEL_40:
              if ( (char *)v29 != v30 )
              {
                v25 = v29;
                while ( *(_QWORD *)v25 != v32 )
                {
                  v25 += 8;
                  if ( (char *)v25 == v30 )
                    goto LABEL_48;
                }
                goto LABEL_44;
              }
LABEL_48:
              v34 = v17[2];
              v33 = v17 + 2;
              goto LABEL_49;
            }
            if ( sub_C8CA60((__int64)&v47, v32) )
              goto LABEL_44;
            v34 = v17[2];
            v33 = v17 + 2;
            if ( v51 )
            {
              v29 = (unsigned __int64)v48;
              v25 = HIDWORD(v49);
              v31 = v48;
              v30 = &v48[8 * HIDWORD(v49)];
LABEL_49:
              if ( (char *)v29 != v30 )
              {
                v25 = v29;
                while ( *(_QWORD *)v25 != v34 )
                {
                  v25 += 8;
                  if ( v30 == (char *)v25 )
                    goto LABEL_57;
                }
                goto LABEL_44;
              }
LABEL_57:
              v35 = v17[3];
              v33 = v17 + 3;
              goto LABEL_58;
            }
            if ( sub_C8CA60((__int64)&v47, v34) )
              goto LABEL_44;
            v35 = v17[3];
            v33 = v17 + 3;
            if ( !v51 )
            {
              if ( sub_C8CA60((__int64)&v47, v35) )
                goto LABEL_44;
              goto LABEL_64;
            }
            v29 = (unsigned __int64)v48;
            v25 = HIDWORD(v49);
            v31 = v48;
            v30 = &v48[8 * HIDWORD(v49)];
LABEL_58:
            if ( v30 != (char *)v29 )
            {
              while ( *(_QWORD *)v31 != v35 )
              {
                v31 += 8;
                if ( v31 == v30 )
                  goto LABEL_64;
              }
LABEL_44:
              v18 = v51;
              goto LABEL_45;
            }
LABEL_64:
            v17 += 4;
            if ( v27 == v17 )
            {
              v26 = v24 - v17;
LABEL_66:
              if ( v26 != 2 )
              {
                if ( v26 != 3 )
                {
                  v18 = v51;
                  if ( v26 == 1 )
                    goto LABEL_69;
LABEL_14:
                  if ( v18 )
                    return 0;
                  _libc_free((unsigned __int64)v48);
                  return 0;
                }
                if ( !(unsigned __int8)sub_B19060((__int64)&v47, *v17, v25, 3) )
                {
                  ++v17;
                  goto LABEL_89;
                }
LABEL_99:
                v18 = v51;
                v33 = v17;
LABEL_45:
                if ( v24 == v33 )
                  goto LABEL_14;
                if ( !v18 )
                  _libc_free((unsigned __int64)v48);
                return 3;
              }
LABEL_89:
              v18 = v51;
              v41 = *v17;
              if ( !v51 )
              {
                v45 = sub_C8CA60((__int64)&v47, v41);
                v18 = v51;
                if ( !v45 )
                {
LABEL_98:
                  ++v17;
LABEL_69:
                  v36 = *v17;
                  if ( v18 )
                  {
                    v37 = v48;
                    v38 = HIDWORD(v49);
                    goto LABEL_71;
                  }
                  v46 = sub_C8CA60((__int64)&v47, *v17) == 0;
                  v18 = v51;
                  if ( v46 )
                    goto LABEL_14;
                }
                v33 = v17;
                goto LABEL_45;
              }
              v37 = v48;
              v38 = HIDWORD(v49);
              v42 = &v48[8 * HIDWORD(v49)];
              v43 = v48;
              if ( v48 == v42 )
              {
                v36 = v17[1];
                ++v17;
LABEL_71:
                v39 = &v37[8 * v38];
                if ( v37 == v39 )
                  return 0;
                while ( *(_QWORD *)v37 != v36 )
                {
                  v37 += 8;
                  if ( v39 == v37 )
                    return 0;
                }
                if ( v24 == v17 )
                  return 0;
                return 3;
              }
              while ( v41 != *(_QWORD *)v43 )
              {
                v43 += 8;
                if ( v42 == v43 )
                  goto LABEL_98;
              }
LABEL_37:
              if ( v17 == v24 )
                return 0;
              return 3;
            }
          }
          v29 = (unsigned __int64)v48;
          v25 = HIDWORD(v49);
          v30 = &v48[8 * HIDWORD(v49)];
          v31 = v48;
          if ( v48 != v30 )
          {
            v25 = (__int64)v48;
            while ( v28 != *(_QWORD *)v25 )
            {
              v25 += 8;
              if ( v30 == (char *)v25 )
                goto LABEL_39;
            }
            goto LABEL_37;
          }
LABEL_39:
          v32 = v17[1];
          v33 = v17 + 1;
          goto LABEL_40;
        }
      }
      else
      {
LABEL_11:
        sub_C8CC70((__int64)&v47, v20, v11, v4, v5, v3);
        v15 = v51;
LABEL_12:
        if ( v19 == ++v12 )
          goto LABEL_13;
      }
    }
  }
  return result;
}

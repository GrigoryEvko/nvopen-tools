// Function: sub_23ED320
// Address: 0x23ed320
//
_BYTE *__fastcall sub_23ED320(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  _BYTE *result; // rax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  unsigned __int8 **v18; // r13
  unsigned __int8 **v19; // r15
  __int64 v20; // r9
  int v21; // r11d
  unsigned __int8 **v22; // rcx
  unsigned int v23; // r8d
  _QWORD *v24; // rax
  unsigned __int8 *v25; // rdi
  unsigned __int8 *v26; // rax
  unsigned __int8 *v27; // rbx
  int v28; // edx
  int v29; // edx
  __int64 v30; // r8
  int v31; // eax
  unsigned __int8 *v32; // rdi
  int v33; // eax
  int v34; // esi
  int v35; // r10d
  unsigned __int8 **v36; // r9
  __int64 v37; // r8
  unsigned int v38; // edx
  unsigned __int8 *v39; // rdi
  int v40; // r10d
  unsigned int v41; // [rsp+4h] [rbp-3Ch]
  __int64 v42; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = (__int64)"llvm.localescape";
  result = sub_BA8CB0(v4, (__int64)"llvm.localescape", 0x10u);
  if ( result )
  {
    v7 = *(_QWORD *)(a2 + 80);
    if ( !v7 )
      BUG();
    v8 = *(_QWORD *)(v7 + 32);
    result = (_BYTE *)(v7 + 24);
    if ( (_BYTE *)v8 != result )
    {
      while ( 1 )
      {
        if ( !v8 )
          BUG();
        if ( *(_BYTE *)(v8 - 24) == 85 )
        {
          v9 = *(_QWORD *)(v8 - 56);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9
              && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v8 + 56)
              && (*(_BYTE *)(v9 + 33) & 0x20) != 0
              && *(_DWORD *)(v9 + 36) == 216 )
            {
              break;
            }
          }
        }
        v8 = *(_QWORD *)(v8 + 8);
        if ( result == (_BYTE *)v8 )
          return result;
      }
      v10 = v8 - 24;
      if ( *(char *)(v8 - 17) < 0 )
      {
        v11 = sub_BD2BC0(v8 - 24);
        v13 = v11 + v12;
        if ( *(char *)(v8 - 17) >= 0 )
        {
          if ( (unsigned int)(v13 >> 4) )
            goto LABEL_66;
        }
        else if ( (unsigned int)((v13 - sub_BD2BC0(v8 - 24)) >> 4) )
        {
          if ( *(char *)(v8 - 17) < 0 )
          {
            v14 = *(_DWORD *)(sub_BD2BC0(v8 - 24) + 8);
            if ( *(char *)(v8 - 17) >= 0 )
              BUG();
            v15 = sub_BD2BC0(v8 - 24);
            v17 = -32 - 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
LABEL_19:
            v18 = (unsigned __int8 **)(v10 + v17);
            v42 = a1 + 1032;
            result = (_BYTE *)(32LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF));
            v19 = (unsigned __int8 **)(v10 - (_QWORD)result);
            if ( v19 == v18 )
              return result;
            while ( 1 )
            {
              v26 = sub_BD3990(*v19, v5);
              v5 = *(unsigned int *)(a1 + 1056);
              v27 = v26;
              if ( *v26 != 60 )
                v27 = 0;
              if ( !(_DWORD)v5 )
                break;
              v20 = *(_QWORD *)(a1 + 1040);
              v21 = 1;
              v22 = 0;
              v23 = (v5 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
              v24 = (_QWORD *)(v20 + 16LL * v23);
              v25 = (unsigned __int8 *)*v24;
              if ( v27 == (unsigned __int8 *)*v24 )
              {
LABEL_22:
                v19 += 4;
                result = v24 + 1;
                *result = 0;
                if ( v18 == v19 )
                  return result;
              }
              else
              {
                while ( v25 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v22 && v25 == (unsigned __int8 *)-8192LL )
                    v22 = (unsigned __int8 **)v24;
                  v23 = (v5 - 1) & (v21 + v23);
                  v24 = (_QWORD *)(v20 + 16LL * v23);
                  v25 = (unsigned __int8 *)*v24;
                  if ( v27 == (unsigned __int8 *)*v24 )
                    goto LABEL_22;
                  ++v21;
                }
                if ( !v22 )
                  v22 = (unsigned __int8 **)v24;
                v33 = *(_DWORD *)(a1 + 1048);
                ++*(_QWORD *)(a1 + 1032);
                v31 = v33 + 1;
                if ( 4 * v31 < (unsigned int)(3 * v5) )
                {
                  if ( (int)v5 - *(_DWORD *)(a1 + 1052) - v31 > (unsigned int)v5 >> 3 )
                    goto LABEL_29;
                  v41 = ((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4);
                  sub_23EC970(v42, v5);
                  v34 = *(_DWORD *)(a1 + 1056);
                  if ( !v34 )
                  {
LABEL_64:
                    ++*(_DWORD *)(a1 + 1048);
                    BUG();
                  }
                  v5 = (unsigned int)(v34 - 1);
                  v35 = 1;
                  v36 = 0;
                  v37 = *(_QWORD *)(a1 + 1040);
                  v38 = v5 & v41;
                  v31 = *(_DWORD *)(a1 + 1048) + 1;
                  v22 = (unsigned __int8 **)(v37 + 16LL * ((unsigned int)v5 & v41));
                  v39 = *v22;
                  if ( v27 == *v22 )
                    goto LABEL_29;
                  while ( v39 != (unsigned __int8 *)-4096LL )
                  {
                    if ( !v36 && v39 == (unsigned __int8 *)-8192LL )
                      v36 = v22;
                    v38 = v5 & (v35 + v38);
                    v22 = (unsigned __int8 **)(v37 + 16LL * v38);
                    v39 = *v22;
                    if ( v27 == *v22 )
                      goto LABEL_29;
                    ++v35;
                  }
                  goto LABEL_48;
                }
LABEL_27:
                sub_23EC970(v42, 2 * v5);
                v28 = *(_DWORD *)(a1 + 1056);
                if ( !v28 )
                  goto LABEL_64;
                v29 = v28 - 1;
                v30 = *(_QWORD *)(a1 + 1040);
                v5 = v29 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
                v31 = *(_DWORD *)(a1 + 1048) + 1;
                v22 = (unsigned __int8 **)(v30 + 16 * v5);
                v32 = *v22;
                if ( v27 == *v22 )
                  goto LABEL_29;
                v40 = 1;
                v36 = 0;
                while ( v32 != (unsigned __int8 *)-4096LL )
                {
                  if ( !v36 && v32 == (unsigned __int8 *)-8192LL )
                    v36 = v22;
                  v5 = v29 & (unsigned int)(v40 + v5);
                  v22 = (unsigned __int8 **)(v30 + 16LL * (unsigned int)v5);
                  v32 = *v22;
                  if ( v27 == *v22 )
                    goto LABEL_29;
                  ++v40;
                }
LABEL_48:
                if ( v36 )
                  v22 = v36;
LABEL_29:
                *(_DWORD *)(a1 + 1048) = v31;
                if ( *v22 != (unsigned __int8 *)-4096LL )
                  --*(_DWORD *)(a1 + 1052);
                result = v22 + 1;
                v19 += 4;
                *v22 = v27;
                *((_BYTE *)v22 + 8) = 0;
                *((_BYTE *)v22 + 8) = 0;
                if ( v18 == v19 )
                  return result;
              }
            }
            ++*(_QWORD *)(a1 + 1032);
            goto LABEL_27;
          }
LABEL_66:
          BUG();
        }
      }
      v17 = -32;
      goto LABEL_19;
    }
  }
  return result;
}

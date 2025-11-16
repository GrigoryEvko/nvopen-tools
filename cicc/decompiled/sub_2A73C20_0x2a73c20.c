// Function: sub_2A73C20
// Address: 0x2a73c20
//
unsigned __int64 *__fastcall sub_2A73C20(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 *v5; // rax
  __int64 v6; // r14
  __int64 *v7; // r13
  __int64 v8; // rax
  _BYTE *v9; // rax
  char v10; // al
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // r9
  __int64 v15; // r13
  bool v16; // r14
  unsigned __int64 *result; // rax
  unsigned __int64 v18; // rbx
  bool v19; // si
  unsigned __int64 v20; // rdi
  int v21; // edx
  __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  __int64 v24; // rdi
  unsigned int i; // eax
  __int64 v26; // rdx
  __int64 v27; // r11
  int v28; // eax
  __int64 v29; // rax
  const __m128i *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r9
  const void *v33; // rsi
  char *v34; // rbx
  bool v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v38; // [rsp+18h] [rbp-98h]
  unsigned __int64 v39; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-88h]
  unsigned __int64 v41; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+38h] [rbp-78h]
  char v43; // [rsp+40h] [rbp-70h]
  unsigned __int64 v44; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v45; // [rsp+58h] [rbp-58h]
  __int64 v46; // [rsp+60h] [rbp-50h]
  _QWORD v47[2]; // [rsp+68h] [rbp-48h] BYREF
  unsigned __int64 v48; // [rsp+78h] [rbp-38h]

  v5 = sub_DD8400(*(_QWORD *)(a1 + 32), a2);
  v6 = *(_QWORD *)(a1 + 32);
  v7 = v5;
  v8 = sub_D95540((__int64)v5);
  v9 = sub_DA2C50(v6, v8, 0, 0);
  v10 = sub_DC3A60(v6, 39, v7, v9);
  v15 = *(_QWORD *)(a2 + 16);
  v16 = v10;
  v38 = a1 + 96;
  result = &v44;
  if ( v15 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v15 + 24);
      if ( !*(_BYTE *)(a1 + 124) )
        goto LABEL_9;
      result = *(unsigned __int64 **)(a1 + 104);
      v12 = *(unsigned int *)(a1 + 116);
      v11 = (unsigned __int64)&result[v12];
      if ( result != (unsigned __int64 *)v11 )
      {
        while ( v18 != *result )
        {
          if ( (unsigned __int64 *)v11 == ++result )
            goto LABEL_16;
        }
        goto LABEL_7;
      }
LABEL_16:
      if ( (unsigned int)v12 < *(_DWORD *)(a1 + 112) )
      {
        v19 = v16;
        *(_DWORD *)(a1 + 116) = v12 + 1;
        *(_QWORD *)v11 = v18;
        ++*(_QWORD *)(a1 + 96);
        if ( !v16 )
        {
LABEL_18:
          v44 = 0;
          v45 = 0;
          v46 = a2;
          if ( a2 != -8192 && a2 != -4096 )
            sub_BD73F0((__int64)&v44);
          v47[0] = 0;
          v47[1] = 0;
          v48 = v18;
          if ( v18 == 0 || v18 == -4096 || v18 == -8192 )
          {
            v22 = *(unsigned int *)(a1 + 312);
            v23 = v18;
            if ( (_DWORD)v22 )
              goto LABEL_24;
          }
          else
          {
            sub_BD73F0((__int64)v47);
            v22 = *(unsigned int *)(a1 + 312);
            v23 = v48;
            if ( (_DWORD)v22 )
            {
LABEL_24:
              v24 = *(_QWORD *)(a1 + 296);
              v13 = 1;
              for ( i = (v22 - 1)
                      & (((0xBF58476D1CE4E5B9LL
                         * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                          | ((unsigned __int64)(((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4)) << 32))) >> 31)
                       ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; i = (v22 - 1) & v28 )
              {
                v26 = v24 + 80LL * i;
                v27 = *(_QWORD *)(v26 + 16);
                if ( v27 == v46 && v23 == *(_QWORD *)(v26 + 40) )
                  break;
                if ( v27 == -4096 && *(_QWORD *)(v26 + 40) == -4096 )
                  goto LABEL_29;
                v28 = v13 + i;
                v13 = (unsigned int)(v13 + 1);
              }
              if ( v26 != 80 * v22 + v24 )
              {
                v40 = *(_DWORD *)(v26 + 56);
                if ( v40 > 0x40 )
                {
                  v36 = v24 + 80LL * i;
                  sub_C43780((__int64)&v39, (const void **)(v26 + 48));
                  v26 = v36;
                  v42 = *(_DWORD *)(v36 + 72);
                  if ( v42 > 0x40 )
                    goto LABEL_58;
LABEL_55:
                  v41 = *(_QWORD *)(v26 + 64);
                }
                else
                {
                  v39 = *(_QWORD *)(v26 + 48);
                  v42 = *(_DWORD *)(v26 + 72);
                  if ( v42 <= 0x40 )
                    goto LABEL_55;
LABEL_58:
                  sub_C43780((__int64)&v41, (const void **)(v26 + 64));
                }
                v43 = 1;
                v23 = v48;
LABEL_30:
                if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
                  sub_BD60C0(v47);
                if ( v46 != 0 && v46 != -4096 && v46 != -8192 )
                  sub_BD60C0(&v44);
                v19 = v43;
                if ( v43 )
                {
                  sub_AB14C0((__int64)&v44, (__int64)&v39);
                  v29 = 1LL << ((unsigned __int8)v45 - 1);
                  if ( (unsigned int)v45 > 0x40 )
                  {
                    v31 = *(_QWORD *)(v44 + 8LL * ((unsigned int)(v45 - 1) >> 6)) & v29;
                    v19 = v31 == 0;
                    if ( v44 )
                    {
                      v35 = v31 == 0;
                      j_j___libc_free_0_0(v44);
                      v19 = v35;
                    }
                  }
                  else
                  {
                    v19 = (v44 & v29) == 0;
                  }
                  if ( v43 )
                  {
                    v43 = 0;
                    if ( v42 > 0x40 && v41 )
                      j_j___libc_free_0_0(v41);
                    if ( v40 > 0x40 && v39 )
                      j_j___libc_free_0_0(v39);
                  }
                }
                goto LABEL_11;
              }
            }
          }
LABEL_29:
          v43 = 0;
          goto LABEL_30;
        }
LABEL_11:
        v12 = *(unsigned int *)(a1 + 328);
        v20 = *(_QWORD *)(a1 + 320);
        v14 = *(unsigned int *)(a1 + 332);
        v21 = *(_DWORD *)(a1 + 328);
        result = (unsigned __int64 *)(v20 + 32 * v12);
        if ( v12 >= v14 )
        {
          v45 = v18;
          v11 = v12 + 1;
          v44 = a2;
          v46 = a3;
          v30 = (const __m128i *)&v44;
          LOBYTE(v47[0]) = v19;
          if ( v14 < v12 + 1 )
          {
            v32 = a1 + 320;
            v33 = (const void *)(a1 + 336);
            if ( v20 > (unsigned __int64)&v44 || result <= &v44 )
            {
              result = (unsigned __int64 *)sub_C8D5F0(a1 + 320, v33, v11, 0x20u, v13, v32);
              v20 = *(_QWORD *)(a1 + 320);
              v12 = *(unsigned int *)(a1 + 328);
              v30 = (const __m128i *)&v44;
            }
            else
            {
              v34 = (char *)&v44 - v20;
              result = (unsigned __int64 *)sub_C8D5F0(a1 + 320, v33, v11, 0x20u, v13, v32);
              v20 = *(_QWORD *)(a1 + 320);
              v12 = *(unsigned int *)(a1 + 328);
              v30 = (const __m128i *)&v34[v20];
            }
          }
          v12 = v20 + 32 * v12;
          *(__m128i *)v12 = _mm_loadu_si128(v30);
          *(__m128i *)(v12 + 16) = _mm_loadu_si128(v30 + 1);
          ++*(_DWORD *)(a1 + 328);
          v15 = *(_QWORD *)(v15 + 8);
          if ( !v15 )
            return result;
        }
        else
        {
          if ( result )
          {
            *result = a2;
            result[1] = v18;
            result[2] = a3;
            *((_BYTE *)result + 24) = v19;
            v21 = *(_DWORD *)(a1 + 328);
          }
          v11 = (unsigned int)(v21 + 1);
          *(_DWORD *)(a1 + 328) = v11;
          v15 = *(_QWORD *)(v15 + 8);
          if ( !v15 )
            return result;
        }
      }
      else
      {
LABEL_9:
        result = sub_C8CC70(v38, *(_QWORD *)(v15 + 24), v11, v12, v13, v14);
        if ( (_BYTE)v11 )
        {
          v19 = v16;
          if ( !v16 )
            goto LABEL_18;
          goto LABEL_11;
        }
LABEL_7:
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          return result;
      }
    }
  }
  return result;
}

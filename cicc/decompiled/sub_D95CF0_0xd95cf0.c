// Function: sub_D95CF0
// Address: 0xd95cf0
//
__int64 *__fastcall sub_D95CF0(_QWORD *a1, unsigned __int8 a2)
{
  __int64 v2; // rdx
  __int64 *result; // rax
  _QWORD *v5; // r10
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 *v9; // r15
  __int64 *v10; // r9
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 *v13; // rax
  __int64 v14; // r15
  __int64 *v15; // r8
  __int64 v16; // r12
  __int64 *v17; // rbx
  bool v18; // zf
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 *v22; // rdx
  __int64 *v23; // r15
  __int64 v24; // r13
  __int64 v25; // rax
  void *v26; // rax
  char *v27; // rax
  __int64 v28; // [rsp-70h] [rbp-70h]
  __int64 *v29; // [rsp-68h] [rbp-68h]
  _QWORD *v30; // [rsp-60h] [rbp-60h]
  __int64 *v31; // [rsp-58h] [rbp-58h]
  unsigned int v32; // [rsp-58h] [rbp-58h]
  __int64 v33; // [rsp-50h] [rbp-50h]
  _QWORD v34[2]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v35; // [rsp-38h] [rbp-38h] BYREF

  v2 = *a1 + 680LL;
  result = (__int64 *)(*a1 + 648LL);
  if ( !a2 )
    v2 = *a1 + 648LL;
  if ( *(_DWORD *)(v2 + 16) )
  {
    v5 = a1;
    result = *(__int64 **)(v2 + 8);
    v6 = *(unsigned int *)(v2 + 24);
    v7 = 5 * v6;
    v8 = (__int64)&result[21 * v6];
    v28 = v8;
    if ( result != (__int64 *)v8 )
    {
      while ( 1 )
      {
        v9 = result;
        if ( *result != -4096 && *result != -8192 )
          break;
        result += 21;
        if ( (__int64 *)v8 == result )
          return result;
      }
      if ( (__int64 *)v8 != result )
      {
        v10 = v34;
        v33 = 4LL * a2;
LABEL_11:
        result = (__int64 *)*((unsigned int *)v9 + 4);
        v11 = v9[1];
        v12 = v11 + 112LL * (_QWORD)result;
        if ( v12 == v11 )
          goto LABEL_32;
        v13 = v9;
        v14 = v12;
        v15 = v13;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v11 + 40);
          result = *(__int64 **)(v11 + 56);
          v17 = v10;
          v18 = *(_WORD *)(v16 + 24) == 0;
          v34[0] = v16;
          v34[1] = result;
          if ( !v18 )
            goto LABEL_16;
          while ( &v35 != ++v17 )
          {
            v16 = *v17;
            if ( *(_WORD *)(*v17 + 24) )
            {
LABEL_16:
              v19 = *(_QWORD *)(*v5 + 720LL);
              v20 = *(unsigned int *)(*v5 + 736LL);
              if ( !(_DWORD)v20 )
                goto LABEL_23;
              v21 = (v20 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v8 = v19 + 72LL * v21;
              v7 = *(_QWORD *)v8;
              if ( *(_QWORD *)v8 != v16 )
              {
                v8 = 1;
                while ( v7 != -4096 )
                {
                  v21 = (v20 - 1) & (v8 + v21);
                  v32 = v8 + 1;
                  v8 = v19 + 72LL * v21;
                  v7 = *(_QWORD *)v8;
                  if ( *(_QWORD *)v8 == v16 )
                    goto LABEL_18;
                  v8 = v32;
                }
LABEL_23:
                v23 = v15;
                v24 = sub_C5F790(v8, v7);
                v25 = *(_QWORD *)(v24 + 32);
                if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) <= 5 )
                {
                  v24 = sub_CB6200(v24, (unsigned __int8 *)"Value ", 6u);
                }
                else
                {
                  *(_DWORD *)v25 = 1970037078;
                  *(_WORD *)(v25 + 4) = 8293;
                  *(_QWORD *)(v24 + 32) += 6LL;
                }
                sub_D955C0(v16, v24);
                v26 = *(void **)(v24 + 32);
                if ( *(_QWORD *)(v24 + 24) - (_QWORD)v26 <= 9u )
                {
                  v24 = sub_CB6200(v24, (unsigned __int8 *)" for loop ", 0xAu);
                }
                else
                {
                  qmemcpy(v26, " for loop ", 10);
                  *(_QWORD *)(v24 + 32) += 10LL;
                }
                sub_D49BF0(*v23, v24, 0, 1, 0);
                v27 = *(char **)(v24 + 32);
                if ( *(_QWORD *)(v24 + 24) - (_QWORD)v27 > 0x1Au )
                {
                  qmemcpy(v27, " missing from BECountUsers\n", 0x1Bu);
                  *(_QWORD *)(v24 + 32) += 27LL;
                }
                else
                {
                  sub_CB6200(v24, " missing from BECountUsers\n", 0x1Bu);
                }
                abort();
              }
LABEL_18:
              if ( v8 == v19 + 72 * v20 )
                goto LABEL_23;
              v7 = v33 | *v15 & 0xFFFFFFFFFFFFFFFBLL;
              if ( *(_BYTE *)(v8 + 36) )
              {
                result = *(__int64 **)(v8 + 16);
                v22 = &result[*(unsigned int *)(v8 + 28)];
                if ( result == v22 )
                  goto LABEL_23;
                while ( v7 != *result )
                {
                  if ( v22 == ++result )
                    goto LABEL_23;
                }
              }
              else
              {
                v8 += 8;
                v29 = v10;
                v30 = v5;
                v31 = v15;
                result = sub_C8CA60(v8, v7);
                v15 = v31;
                v5 = v30;
                v10 = v29;
                if ( !result )
                  goto LABEL_23;
              }
            }
          }
          v11 += 112;
          if ( v11 == v14 )
          {
            v9 = v15;
LABEL_32:
            v9 += 21;
            if ( v9 == (__int64 *)v28 )
              return result;
            while ( 1 )
            {
              result = (__int64 *)*v9;
              if ( *v9 != -4096 && result != (__int64 *)-8192LL )
                break;
              v9 += 21;
              if ( (__int64 *)v28 == v9 )
                return result;
            }
            if ( (__int64 *)v28 == v9 )
              return result;
            goto LABEL_11;
          }
        }
      }
    }
  }
  return result;
}

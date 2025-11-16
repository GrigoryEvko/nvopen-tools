// Function: sub_ADF3C0
// Address: 0xadf3c0
//
_QWORD *__fastcall sub_ADF3C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 **v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 **v8; // r13
  _QWORD *i; // rdx
  __int64 **v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // r8
  int v15; // r9d
  unsigned int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // r14
  __int64 *v19; // rsi
  __int64 *v20; // r9
  unsigned int v21; // r10d
  __int64 **v22; // r15
  __int64 **v23; // r14
  __int64 **v24; // r15
  __int64 v25; // rdx
  __int64 **v26; // rcx
  __int64 **v27; // r8
  __int64 **v28; // r13
  __int64 **v29; // rbx
  __int64 **v30; // r12
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdi
  int v40; // edx
  __int64 *v41; // rax
  _QWORD *j; // rdx
  __int64 v43; // [rsp+0h] [rbp-70h]
  __int64 v44; // [rsp+8h] [rbp-68h]
  unsigned int v45; // [rsp+8h] [rbp-68h]
  unsigned int v46; // [rsp+10h] [rbp-60h]
  int v47; // [rsp+10h] [rbp-60h]
  __int64 **v48; // [rsp+18h] [rbp-58h]
  __int64 **v49; // [rsp+20h] [rbp-50h]
  __int64 **v50; // [rsp+20h] [rbp-50h]
  __int64 **v51; // [rsp+28h] [rbp-48h]
  __int64 v52[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = a1;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(__int64 ***)(a1 + 8);
  v51 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v43 = 56 * v4;
    v8 = &v5[7 * v4];
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = v5 + 3;
    if ( v8 != v5 )
    {
      while ( 1 )
      {
        v11 = (__int64)*(v10 - 3);
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(v3 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 3);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(v3 + 8);
          v15 = 1;
          v16 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v17 = 0;
          v18 = v14 + 56LL * v16;
          v19 = *(__int64 **)v18;
          if ( v11 != *(_QWORD *)v18 )
          {
            while ( v19 != (__int64 *)-4096LL )
            {
              if ( !v17 && v19 == (__int64 *)-8192LL )
                v17 = v18;
              v16 = v13 & (v15 + v16);
              v18 = v14 + 56LL * v16;
              v19 = *(__int64 **)v18;
              if ( v11 == *(_QWORD *)v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v17 )
              v18 = v17;
          }
LABEL_13:
          *(_QWORD *)v18 = v11;
          v20 = (__int64 *)(v18 + 24);
          *(_QWORD *)(v18 + 16) = 0x400000000LL;
          *(_QWORD *)(v18 + 8) = v18 + 24;
          v21 = *((_DWORD *)v10 - 2);
          if ( (__int64 **)(v18 + 8) != v10 - 2 && v21 )
          {
            v24 = (__int64 **)*(v10 - 2);
            if ( v10 == v24 )
            {
              v25 = v21;
              v26 = v10;
              if ( v21 > 4 )
              {
                v45 = *((_DWORD *)v10 - 2);
                v19 = (__int64 *)sub_C8D7D0(v18 + 8, v18 + 24, v21, 8, v52);
                sub_ADDB20(v18 + 8, v19, v35, v36, v37, v38);
                v39 = *(_QWORD *)(v18 + 8);
                v40 = v52[0];
                v41 = v19;
                v21 = v45;
                if ( v18 + 24 != v39 )
                {
                  v47 = v52[0];
                  _libc_free(v39, v19);
                  v40 = v47;
                  v41 = v19;
                  v21 = v45;
                }
                *(_QWORD *)(v18 + 8) = v41;
                v20 = v41;
                *(_DWORD *)(v18 + 20) = v40;
                v26 = (__int64 **)*(v24 - 2);
                v25 = *((unsigned int *)v24 - 2);
              }
              v27 = &v26[v25];
              if ( v27 != v26 )
              {
                v46 = v21;
                v49 = v8;
                v28 = v26;
                v48 = v10;
                v29 = (__int64 **)v20;
                v44 = v3;
                v30 = &v26[v25];
                do
                {
                  if ( v29 )
                  {
                    v19 = *v28;
                    *v29 = *v28;
                    if ( v19 )
                    {
                      sub_B976B0(v28, v19, v29, v26, v27, v20);
                      *v28 = 0;
                    }
                  }
                  ++v28;
                  ++v29;
                }
                while ( v30 != v28 );
                v8 = v49;
                v10 = v48;
                v21 = v46;
                v3 = v44;
              }
              *(_DWORD *)(v18 + 16) = v21;
              v31 = (__int64)*(v24 - 2);
              v32 = *((unsigned int *)v24 - 2);
              if ( v31 != v31 + 8 * v32 )
              {
                v50 = v10;
                v33 = v31 + 8 * v32;
                v34 = (__int64)*(v24 - 2);
                do
                {
                  v19 = *(__int64 **)(v33 - 8);
                  v33 -= 8;
                  if ( v19 )
                    sub_B91220(v33);
                }
                while ( v34 != v33 );
                v10 = v50;
              }
              *((_DWORD *)v24 - 2) = 0;
            }
            else
            {
              *(_QWORD *)(v18 + 8) = v24;
              *(_DWORD *)(v18 + 16) = *((_DWORD *)v10 - 2);
              *(_DWORD *)(v18 + 20) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = (__int64 *)v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(v3 + 16);
          v22 = (__int64 **)*(v10 - 2);
          v23 = &v22[*((unsigned int *)v10 - 2)];
          if ( v22 != v23 )
          {
            do
            {
              v19 = *--v23;
              if ( v19 )
                sub_B91220(v23);
            }
            while ( v22 != v23 );
            v23 = (__int64 **)*(v10 - 2);
          }
          if ( v10 != v23 )
            _libc_free(v23, v19);
        }
        if ( v8 == v10 + 4 )
          break;
        v10 += 7;
      }
    }
    return (_QWORD *)sub_C7D6A0(v51, v43, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}

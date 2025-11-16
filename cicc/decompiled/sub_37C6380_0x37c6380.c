// Function: sub_37C6380
// Address: 0x37c6380
//
_DWORD *__fastcall sub_37C6380(__int64 a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // rcx
  unsigned __int64 *v8; // r13
  _DWORD *i; // rdx
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // rax
  unsigned int v12; // eax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  __int64 v16; // r8
  int v17; // r9d
  unsigned int v18; // ecx
  __int64 v19; // r14
  int v20; // esi
  __int64 v21; // r15
  __int64 v22; // r9
  unsigned __int64 *v23; // r15
  unsigned __int64 *v24; // r14
  unsigned __int64 *v25; // rax
  unsigned __int64 *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  unsigned __int64 *v29; // r13
  __int64 v30; // rbx
  int v31; // r15d
  unsigned __int64 *v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned __int64 v37; // rdi
  int v38; // eax
  __int64 v39; // rdx
  unsigned __int64 *v40; // r15
  unsigned __int64 *v41; // r14
  _DWORD *j; // rdx
  __int64 v43; // [rsp+8h] [rbp-68h]
  unsigned int v44; // [rsp+8h] [rbp-68h]
  int v45; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v46; // [rsp+10h] [rbp-60h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v48; // [rsp+18h] [rbp-58h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  unsigned int v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+28h] [rbp-48h]
  unsigned __int64 v53[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v52 = v3;
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(112LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v51 = 112 * v4;
    v8 = (unsigned __int64 *)(112 * v4 + v3);
    for ( i = &result[28 * v7]; i != result; result += 28 )
    {
      if ( result )
        *result = -1;
    }
    v10 = (unsigned __int64 *)(v3 + 24);
    if ( v8 != (unsigned __int64 *)v3 )
    {
      while ( 1 )
      {
        v12 = *((_DWORD *)v10 - 6);
        if ( v12 > 0xFFFFFFFD )
          goto LABEL_10;
        v13 = *(_DWORD *)(v2 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(v2 + 8);
        v16 = 0;
        v17 = 1;
        v18 = v14 & (37 * v12);
        v19 = v15 + 112LL * v18;
        v20 = *(_DWORD *)v19;
        if ( v12 != *(_DWORD *)v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v16 && v20 == -2 )
              v16 = v19;
            v18 = v14 & (v17 + v18);
            v19 = v15 + 112LL * v18;
            v20 = *(_DWORD *)v19;
            if ( v12 == *(_DWORD *)v19 )
              goto LABEL_15;
            ++v17;
          }
          if ( v16 )
            v19 = v16;
        }
LABEL_15:
        *(_DWORD *)v19 = v12;
        v21 = v19 + 24;
        *(_QWORD *)(v19 + 16) = 0x100000000LL;
        *(_QWORD *)(v19 + 8) = v19 + 24;
        v22 = *((unsigned int *)v10 - 2);
        if ( (unsigned __int64 *)(v19 + 8) != v10 - 2 && (_DWORD)v22 )
        {
          v25 = (unsigned __int64 *)*(v10 - 2);
          if ( v25 == v10 )
          {
            v26 = v10;
            v27 = 1;
            if ( (_DWORD)v22 != 1 )
            {
              v44 = *((_DWORD *)v10 - 2);
              v49 = sub_C8D7D0(v19 + 8, v19 + 24, (unsigned int)v22, 0x58u, v53, v22);
              sub_37BF480(v19 + 8, v49, v33, v34, v35, v36);
              v37 = *(_QWORD *)(v19 + 8);
              v38 = v53[0];
              v39 = v49;
              v22 = v44;
              if ( v21 != v37 )
              {
                v45 = v53[0];
                v47 = v49;
                v50 = v22;
                _libc_free(v37);
                v38 = v45;
                v39 = v47;
                v22 = v50;
              }
              *(_QWORD *)(v19 + 8) = v39;
              v21 = v39;
              *(_DWORD *)(v19 + 20) = v38;
              v26 = (unsigned __int64 *)*(v10 - 2);
              v27 = *((unsigned int *)v10 - 2);
            }
            v28 = (__int64)&v26[11 * v27];
            if ( (unsigned __int64 *)v28 != v26 )
            {
              v48 = v8;
              v29 = v26;
              v46 = v10;
              v30 = v21;
              v31 = v22;
              v43 = v2;
              v32 = &v26[11 * v27];
              do
              {
                if ( v30 )
                {
                  *(_DWORD *)(v30 + 8) = 0;
                  *(_QWORD *)v30 = v30 + 16;
                  *(_DWORD *)(v30 + 12) = 1;
                  if ( *((_DWORD *)v29 + 2) )
                    sub_37B6100(v30, (char **)v29, v28, v27, v16, v22);
                  *(_DWORD *)(v30 + 64) = *((_DWORD *)v29 + 16);
                  *(__m128i *)(v30 + 72) = _mm_loadu_si128((const __m128i *)(v29 + 9));
                }
                v29 += 11;
                v30 += 88;
              }
              while ( v32 != v29 );
              v8 = v48;
              v10 = v46;
              LODWORD(v22) = v31;
              v2 = v43;
            }
            *(_DWORD *)(v19 + 16) = v22;
            v40 = (unsigned __int64 *)*(v10 - 2);
            v41 = &v40[11 * *((unsigned int *)v10 - 2)];
            while ( v40 != v41 )
            {
              v41 -= 11;
              if ( (unsigned __int64 *)*v41 != v41 + 2 )
                _libc_free(*v41);
            }
            *((_DWORD *)v10 - 2) = 0;
          }
          else
          {
            *(_QWORD *)(v19 + 8) = v25;
            *(_DWORD *)(v19 + 16) = *((_DWORD *)v10 - 2);
            *(_DWORD *)(v19 + 20) = *((_DWORD *)v10 - 1);
            *(v10 - 2) = (unsigned __int64)v10;
            *((_DWORD *)v10 - 1) = 0;
            *((_DWORD *)v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(v2 + 16);
        v23 = (unsigned __int64 *)*(v10 - 2);
        v24 = &v23[11 * *((unsigned int *)v10 - 2)];
        if ( v23 != v24 )
        {
          do
          {
            v24 -= 11;
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              _libc_free(*v24);
          }
          while ( v23 != v24 );
          v24 = (unsigned __int64 *)*(v10 - 2);
        }
        if ( v10 == v24 )
        {
LABEL_10:
          v11 = v10 + 14;
          if ( v8 == v10 + 11 )
            return (_DWORD *)sub_C7D6A0(v52, v51, 8);
        }
        else
        {
          _libc_free((unsigned __int64)v24);
          v11 = v10 + 14;
          if ( v8 == v10 + 11 )
            return (_DWORD *)sub_C7D6A0(v52, v51, 8);
        }
        v10 = v11;
      }
    }
    return (_DWORD *)sub_C7D6A0(v52, v51, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[28 * *(unsigned int *)(a1 + 24)]; j != result; result += 28 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}

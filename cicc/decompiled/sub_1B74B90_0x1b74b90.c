// Function: sub_1B74B90
// Address: 0x1b74b90
//
__int64 __fastcall sub_1B74B90(
        __int64 a1,
        unsigned int a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax
  char v12; // dl
  __int64 *v13; // r13
  unsigned __int64 v14; // rax
  int v15; // r14d
  __int64 v16; // rdi
  __int64 *v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  _QWORD *v24; // rdi
  __int64 v25; // r8
  int v26; // edi
  int v27; // r11d
  __int64 *v28; // r10
  unsigned int v29; // esi
  __int64 *v30; // rcx
  __int64 v31; // r9
  __int64 v32; // rdx
  int v33; // ecx
  unsigned int v34; // ebx
  bool v35; // zf
  __int64 *v36; // r14
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v41; // rdi
  __int64 v42; // rsi
  int v43; // r10d
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // r8
  __int64 v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rax
  int v51; // edx
  __int64 v52; // rax
  _BYTE v53[816]; // [rsp+10h] [rbp-330h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v12 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x1F )
  {
    if ( v12 )
      return result;
    v13 = *(__int64 **)(a1 + 16);
    v34 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v13 = *(__int64 **)(a1 + 16);
    v14 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
             | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
             | (a2 - 1)
             | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
           | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
           | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
           | (a2 - 1)
           | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
         | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
           | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
           | (a2 - 1)
           | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
         | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
         | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
         | (a2 - 1)
         | ((unsigned __int64)(a2 - 1) >> 1))
        + 1;
    v15 = v14;
    if ( (unsigned int)v14 > 0x40 )
    {
      v16 = 3LL * (unsigned int)v14;
      if ( v12 )
        goto LABEL_5;
      v34 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v12 )
      {
        v16 = 192;
        v15 = 64;
LABEL_5:
        v17 = (__int64 *)(a1 + 16);
        v18 = v53;
        do
        {
          v19 = *v17;
          if ( *v17 != -4 && v19 != -8 )
          {
            if ( v18 )
              *v18 = v19;
            v18 += 3;
            *((_BYTE *)v18 - 16) = *((_BYTE *)v17 + 8);
            *((_DWORD *)v18 - 3) = *((_DWORD *)v17 + 3);
            *(v18 - 1) = v17[2];
          }
          v17 += 3;
        }
        while ( v17 != (__int64 *)(a1 + 784) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v20 = (_QWORD *)sub_22077B0(v16 * 8);
        v21 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = v20;
        v22 = v21 & 1;
        *(_DWORD *)(a1 + 24) = v15;
        *(_QWORD *)(a1 + 8) = v22;
        if ( (_BYTE)v22 )
        {
          v20 = (_QWORD *)(a1 + 16);
          v16 = 96;
        }
        v23 = v20;
        v24 = &v20[v16];
        while ( 1 )
        {
          if ( v23 )
            *v20 = -4;
          v20 += 3;
          if ( v24 == v20 )
            break;
          v23 = v20;
        }
        for ( result = (__int64)v53; v18 != (_QWORD *)result; result += 24 )
        {
          v32 = *(_QWORD *)result;
          if ( *(_QWORD *)result != -4 && v32 != -8 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v25 = a1 + 16;
              v26 = 31;
            }
            else
            {
              v33 = *(_DWORD *)(a1 + 24);
              v25 = *(_QWORD *)(a1 + 16);
              if ( !v33 )
              {
                MEMORY[0] = *(_QWORD *)result;
                BUG();
              }
              v26 = v33 - 1;
            }
            v27 = 1;
            v28 = 0;
            v29 = v26 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v30 = (__int64 *)(v25 + 24LL * v29);
            v31 = *v30;
            if ( v32 != *v30 )
            {
              while ( v31 != -4 )
              {
                if ( v31 == -8 && !v28 )
                  v28 = v30;
                v29 = v26 & (v27 + v29);
                v30 = (__int64 *)(v25 + 24LL * v29);
                v31 = *v30;
                if ( v32 == *v30 )
                  goto LABEL_23;
                ++v27;
              }
              if ( v28 )
                v30 = v28;
            }
LABEL_23:
            *v30 = v32;
            *((_BYTE *)v30 + 8) = *(_BYTE *)(result + 8);
            *((_DWORD *)v30 + 3) = *(_DWORD *)(result + 12);
            v30[2] = *(_QWORD *)(result + 16);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return result;
      }
      v34 = *(_DWORD *)(a1 + 24);
      v15 = 64;
      v16 = 192;
    }
    v52 = sub_22077B0(v16 * 8);
    *(_DWORD *)(a1 + 24) = v15;
    *(_QWORD *)(a1 + 16) = v52;
  }
  v35 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v36 = &v13[3 * v34];
  if ( v35 )
  {
    v37 = *(_QWORD **)(a1 + 16);
    v38 = 3LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v37 = (_QWORD *)(a1 + 16);
    v38 = 96;
  }
  for ( i = &v37[v38]; i != v37; v37 += 3 )
  {
    if ( v37 )
      *v37 = -4;
  }
  for ( j = v13; v36 != j; j += 3 )
  {
    v50 = *j;
    if ( *j != -8 && v50 != -4 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v41 = a1 + 16;
        v42 = 31;
      }
      else
      {
        v51 = *(_DWORD *)(a1 + 24);
        v41 = *(_QWORD *)(a1 + 16);
        if ( !v51 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v42 = (unsigned int)(v51 - 1);
      }
      v43 = 1;
      v44 = 0;
      v45 = (unsigned int)v42 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
      v46 = v41 + 24 * v45;
      v47 = *(_QWORD *)v46;
      if ( *(_QWORD *)v46 != v50 )
      {
        while ( v47 != -4 )
        {
          if ( !v44 && v47 == -8 )
            v44 = v46;
          v45 = (unsigned int)v42 & (v43 + (_DWORD)v45);
          v46 = v41 + 24LL * (unsigned int)v45;
          v47 = *(_QWORD *)v46;
          if ( v50 == *(_QWORD *)v46 )
            goto LABEL_43;
          ++v43;
        }
        if ( v44 )
          v46 = v44;
      }
LABEL_43:
      *(_QWORD *)v46 = v50;
      *(_BYTE *)(v46 + 8) = *((_BYTE *)j + 8);
      *(_DWORD *)(v46 + 12) = *((_DWORD *)j + 3);
      *(_QWORD *)(v46 + 16) = j[2];
      j[2] = 0;
      v48 = (unsigned int)(2 * (*(_DWORD *)(a1 + 8) >> 1) + 2);
      *(_DWORD *)(a1 + 8) = v48 | *(_DWORD *)(a1 + 8) & 1;
      v49 = j[2];
      if ( v49 )
        sub_16307F0(v49, v42, v48, v45, v47, a3, a4, a5, a6, a7, a8, a9, a10);
    }
  }
  return j___libc_free_0(v13);
}

// Function: sub_2C04050
// Address: 0x2c04050
//
__int64 __fastcall sub_2C04050(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // r9
  char v11; // cl
  __int64 v12; // rdi
  int v13; // esi
  unsigned int v14; // edx
  __int64 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r14
  __int64 v19; // rdx
  __int64 *v20; // rbx
  _QWORD *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // rsi
  __int64 v25; // r8
  __int64 v26; // rdx
  _QWORD *v27; // rdi
  _QWORD *v28; // rdx
  _QWORD *v29; // rdx
  size_t v30; // rdx
  _QWORD *v31; // r15
  __int64 v32; // rbx
  const void *v33; // r14
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r12
  int v38; // r8d
  __int64 v39[10]; // [rsp+0h] [rbp-50h] BYREF

  v9 = 1;
  if ( *(_BYTE *)(a3 + 8) )
  {
    v36 = a3;
    while ( 1 )
    {
      v9 = *(unsigned int *)(v36 + 88);
      if ( (_DWORD)v9 )
        break;
      v36 = *(_QWORD *)(v36 + 48);
      if ( !v36 )
      {
        v9 = 0;
        break;
      }
    }
  }
  v39[1] = v9;
  v39[0] = a3;
  v39[2] = a3;
  v39[3] = 0;
  sub_2C03DE0((_QWORD *)a1, v39, v9, a4, a5, a6);
  sub_2C03EE0(a1);
  v11 = *(_BYTE *)(a2 + 8) & 1;
  if ( v11 )
  {
    v12 = a2 + 16;
    v13 = 3;
  }
  else
  {
    v35 = *(unsigned int *)(a2 + 24);
    v12 = *(_QWORD *)(a2 + 16);
    if ( !(_DWORD)v35 )
      goto LABEL_54;
    v13 = v35 - 1;
  }
  v14 = v13 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v12 + 72LL * v14);
  v16 = *v15;
  if ( *v15 == a3 )
    goto LABEL_5;
  v38 = 1;
  while ( v16 != -4096 )
  {
    v10 = (unsigned int)(v38 + 1);
    v14 = v13 & (v38 + v14);
    v15 = (__int64 *)(v12 + 72LL * v14);
    v16 = *v15;
    if ( *v15 == a3 )
      goto LABEL_5;
    ++v38;
  }
  if ( v11 )
  {
    v37 = 288;
    goto LABEL_55;
  }
  v35 = *(unsigned int *)(a2 + 24);
LABEL_54:
  v37 = 72 * v35;
LABEL_55:
  v15 = (__int64 *)(v12 + v37);
LABEL_5:
  v17 = 288;
  if ( !v11 )
    v17 = 72LL * *(unsigned int *)(a2 + 24);
  if ( v15 != (__int64 *)(v12 + v17) )
  {
    v18 = (__int64 *)v15[1];
    v19 = *(unsigned int *)(a1 + 8);
    v20 = &v18[*((unsigned int *)v15 + 4)];
    if ( v20 != v18 )
    {
      v21 = *(_QWORD **)a1;
      while ( 1 )
      {
        v22 = 8 * v19;
        v23 = *v18;
        v24 = &v21[(unsigned __int64)v22 / 8];
        v25 = v22 >> 3;
        v26 = v22 >> 5;
        if ( v26 )
          break;
        v27 = v21;
LABEL_35:
        switch ( v25 )
        {
          case 2LL:
            goto LABEL_41;
          case 3LL:
            if ( v23 != *v27 )
            {
              ++v27;
LABEL_41:
              if ( v23 != *v27 )
              {
                ++v27;
LABEL_43:
                v31 = v24;
                if ( v23 != *v27 )
                  goto LABEL_24;
              }
            }
LABEL_17:
            if ( v24 != v27 )
            {
              v29 = v27 + 1;
              if ( v24 == v27 + 1 )
              {
                v31 = v27;
              }
              else
              {
                do
                {
                  if ( v23 != *v29 )
                    *v27++ = *v29;
                  ++v29;
                }
                while ( v24 != v29 );
                v21 = *(_QWORD **)a1;
                v30 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - (_QWORD)v24;
                v31 = (_QWORD *)((char *)v27 + v30);
                if ( v24 != (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)) )
                {
                  memmove(v27, v24, v30);
                  v21 = *(_QWORD **)a1;
                }
              }
              goto LABEL_24;
            }
            break;
          case 1LL:
            goto LABEL_43;
        }
        v31 = v24;
LABEL_24:
        ++v18;
        *(_DWORD *)(a1 + 8) = v31 - v21;
        v19 = (unsigned int)(v31 - v21);
        if ( v20 == v18 )
          goto LABEL_25;
      }
      v27 = v21;
      v28 = &v21[4 * v26];
      while ( v23 != *v27 )
      {
        if ( v23 == v27[1] )
        {
          ++v27;
          goto LABEL_17;
        }
        if ( v23 == v27[2] )
        {
          v27 += 2;
          goto LABEL_17;
        }
        if ( v23 == v27[3] )
        {
          v27 += 3;
          goto LABEL_17;
        }
        v27 += 4;
        if ( v28 == v27 )
        {
          v25 = v24 - v27;
          goto LABEL_35;
        }
      }
      goto LABEL_17;
    }
LABEL_25:
    v32 = *((unsigned int *)v15 + 12);
    v33 = (const void *)v15[5];
    if ( v19 + v32 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + v32, 8u, v19 + v32, v10);
      v19 = *(unsigned int *)(a1 + 8);
    }
    if ( 8 * v32 )
    {
      memcpy((void *)(*(_QWORD *)a1 + 8 * v19), v33, 8 * v32);
      LODWORD(v19) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v32 + v19;
  }
  return a1;
}

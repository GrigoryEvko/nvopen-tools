// Function: sub_143AD00
// Address: 0x143ad00
//
__int64 __fastcall sub_143AD00(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 *v4; // r14
  unsigned __int64 v5; // r13
  int v6; // r15d
  __int64 v7; // r13
  __int64 *v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r13
  _BYTE *i; // rdx
  __int64 v16; // r8
  int v17; // edi
  int v18; // r13d
  __int64 *v19; // r10
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r9
  __int64 v23; // rcx
  int v24; // edi
  __int64 v25; // rbx
  bool v26; // zf
  __int64 *v27; // rsi
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx
  __int64 *k; // rax
  __int64 v32; // r9
  int v33; // r10d
  int v34; // r13d
  __int64 *v35; // rbx
  unsigned int v36; // edi
  __int64 *v37; // rcx
  __int64 v38; // r11
  __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rax
  _BYTE v42[560]; // [rsp+10h] [rbp-230h] BYREF

  result = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x1F )
  {
    if ( (_BYTE)result )
      return result;
    v4 = *(__int64 **)(a1 + 16);
    v25 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
  }
  else
  {
    v4 = *(__int64 **)(a1 + 16);
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = 16LL * (unsigned int)v5;
      if ( (_BYTE)result )
        goto LABEL_5;
      v25 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( (_BYTE)result )
      {
        v7 = 1024;
        v6 = 64;
LABEL_5:
        v8 = (__int64 *)(a1 + 16);
        v9 = v42;
        do
        {
          v10 = *v8;
          if ( *v8 != -8 && v10 != -16 )
          {
            if ( v9 )
              *v9 = v10;
            v9 += 2;
            *((_DWORD *)v9 - 2) = *((_DWORD *)v8 + 2);
          }
          v8 += 2;
        }
        while ( v8 != (__int64 *)(a1 + 528) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v7);
        v11 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = result;
        v12 = v11 & 1;
        *(_QWORD *)(a1 + 8) = v12;
        if ( (_BYTE)v12 )
        {
          result = a1 + 16;
          v7 = 512;
        }
        v13 = result;
        v14 = result + v7;
        while ( 1 )
        {
          if ( v13 )
            *(_QWORD *)result = -8;
          result += 16;
          if ( v14 == result )
            break;
          v13 = result;
        }
        for ( i = v42; v9 != (_QWORD *)i; i += 16 )
        {
          v23 = *(_QWORD *)i;
          if ( *(_QWORD *)i != -8 && v23 != -16 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v16 = a1 + 16;
              v17 = 31;
            }
            else
            {
              v24 = *(_DWORD *)(a1 + 24);
              v16 = *(_QWORD *)(a1 + 16);
              if ( !v24 )
              {
                MEMORY[0] = *(_QWORD *)i;
                BUG();
              }
              v17 = v24 - 1;
            }
            v18 = 1;
            v19 = 0;
            v20 = v17 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v21 = (__int64 *)(v16 + 16LL * v20);
            v22 = *v21;
            if ( v23 != *v21 )
            {
              while ( v22 != -8 )
              {
                if ( v22 == -16 && !v19 )
                  v19 = v21;
                v20 = v17 & (v18 + v20);
                v21 = (__int64 *)(v16 + 16LL * v20);
                v22 = *v21;
                if ( v23 == *v21 )
                  goto LABEL_23;
                ++v18;
              }
              if ( v19 )
                v21 = v19;
            }
LABEL_23:
            *v21 = v23;
            *((_DWORD *)v21 + 2) = *((_DWORD *)i + 2);
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
        }
        return result;
      }
      v25 = *(unsigned int *)(a1 + 24);
      v7 = 1024;
      v6 = 64;
    }
    v41 = sub_22077B0(v7);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v41;
  }
  v26 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v27 = &v4[2 * v25];
  if ( v26 )
  {
    v28 = *(_QWORD **)(a1 + 16);
    v29 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v28 = (_QWORD *)(a1 + 16);
    v29 = 64;
  }
  for ( j = &v28[v29]; j != v28; v28 += 2 )
  {
    if ( v28 )
      *v28 = -8;
  }
  for ( k = v4; v27 != k; k += 2 )
  {
    v39 = *k;
    if ( *k != -16 && v39 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v32 = a1 + 16;
        v33 = 31;
      }
      else
      {
        v40 = *(_DWORD *)(a1 + 24);
        v32 = *(_QWORD *)(a1 + 16);
        if ( !v40 )
        {
          MEMORY[0] = *k;
          BUG();
        }
        v33 = v40 - 1;
      }
      v34 = 1;
      v35 = 0;
      v36 = v33 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v37 = (__int64 *)(v32 + 16LL * v36);
      v38 = *v37;
      if ( *v37 != v39 )
      {
        while ( v38 != -8 )
        {
          if ( !v35 && v38 == -16 )
            v35 = v37;
          v36 = v33 & (v34 + v36);
          v37 = (__int64 *)(v32 + 16LL * v36);
          v38 = *v37;
          if ( v39 == *v37 )
            goto LABEL_43;
          ++v34;
        }
        if ( v35 )
          v37 = v35;
      }
LABEL_43:
      *v37 = v39;
      *((_DWORD *)v37 + 2) = *((_DWORD *)k + 2);
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v4);
}

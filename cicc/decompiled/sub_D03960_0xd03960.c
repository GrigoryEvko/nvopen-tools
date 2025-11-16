// Function: sub_D03960
// Address: 0xd03960
//
__int64 __fastcall sub_D03960(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  unsigned int v5; // ecx
  unsigned int v6; // r11d
  __int64 v7; // rsi
  int v8; // r11d
  unsigned int v10; // edi
  __int64 *v11; // r8
  __int64 v12; // rcx
  unsigned __int64 v13; // rdi
  __int64 *v14; // rcx
  __int64 v15; // r8
  int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // r9
  int v19; // esi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r12
  unsigned int v23; // r11d
  __int64 *v24; // r12
  __int64 v25; // r13
  int v26; // eax
  int v27; // r13d
  int v28; // r8d
  int v29; // r9d
  int v30; // ecx

  result = *(unsigned int *)(a1 + 80);
  v3 = *(_QWORD *)(a1 + 64);
  if ( !(_DWORD)result )
    return result;
  v5 = (unsigned int)a2 >> 9;
  v6 = (unsigned int)a2 >> 4;
  v7 = (unsigned int)(result - 1);
  v8 = v5 ^ v6;
  v10 = v7 & v8;
  v11 = (__int64 *)(v3 + 16LL * ((unsigned int)v7 & v8));
  v12 = *v11;
  if ( *v11 == a2 )
  {
LABEL_3:
    result = v3 + 16 * result;
    if ( v11 == (__int64 *)result )
      return result;
    v13 = v11[1] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v11[1] & 4) != 0 )
    {
      v14 = *(__int64 **)v13;
      v15 = *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8);
    }
    else
    {
      v14 = v11 + 1;
      if ( !v13 )
      {
LABEL_13:
        v23 = v7 & v8;
        v24 = (__int64 *)(v3 + 16LL * v23);
        result = *v24;
        if ( *v24 == a2 )
        {
LABEL_14:
          result = v24[1];
          if ( result )
          {
            if ( (result & 4) != 0 )
            {
              result &= 0xFFFFFFFFFFFFFFF8LL;
              v25 = result;
              if ( result )
              {
                if ( *(_QWORD *)result != result + 16 )
                  _libc_free(*(_QWORD *)result, v7);
                result = j_j___libc_free_0(v25, 48);
              }
            }
          }
          *v24 = -8192;
          --*(_DWORD *)(a1 + 72);
          ++*(_DWORD *)(a1 + 76);
        }
        else
        {
          v30 = 1;
          while ( result != -4096 )
          {
            v23 = v7 & (v30 + v23);
            v24 = (__int64 *)(v3 + 16LL * v23);
            result = *v24;
            if ( *v24 == a2 )
              goto LABEL_14;
            ++v30;
          }
        }
        return result;
      }
      v15 = (__int64)(v11 + 2);
    }
    if ( (__int64 *)v15 == v14 )
      goto LABEL_13;
    do
    {
      v16 = *(_DWORD *)(a1 + 48);
      v17 = *v14;
      v18 = *(_QWORD *)(a1 + 32);
      if ( v16 )
      {
        v19 = v16 - 1;
        v20 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( v17 == *v21 )
        {
LABEL_9:
          *v21 = -8192;
          --*(_DWORD *)(a1 + 40);
          ++*(_DWORD *)(a1 + 44);
        }
        else
        {
          v26 = 1;
          while ( v22 != -4096 )
          {
            v27 = v26 + 1;
            v20 = v19 & (v26 + v20);
            v21 = (__int64 *)(v18 + 16LL * v20);
            v22 = *v21;
            if ( v17 == *v21 )
              goto LABEL_9;
            v26 = v27;
          }
        }
      }
      ++v14;
    }
    while ( v14 != (__int64 *)v15 );
    result = *(unsigned int *)(a1 + 80);
    v3 = *(_QWORD *)(a1 + 64);
    if ( (_DWORD)result )
    {
      v7 = (unsigned int)(result - 1);
      goto LABEL_13;
    }
  }
  else
  {
    v28 = 1;
    while ( v12 != -4096 )
    {
      v29 = v28 + 1;
      v10 = v7 & (v28 + v10);
      v11 = (__int64 *)(v3 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == a2 )
        goto LABEL_3;
      v28 = v29;
    }
  }
  return result;
}

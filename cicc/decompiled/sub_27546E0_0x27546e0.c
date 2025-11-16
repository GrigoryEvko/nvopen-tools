// Function: sub_27546E0
// Address: 0x27546e0
//
__int64 __fastcall sub_27546E0(__int64 a1, _QWORD *a2)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // r13
  _QWORD *v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 result; // rax
  unsigned __int64 v23; // r14
  _QWORD *v24; // rdx
  _QWORD *v25; // rcx
  unsigned __int64 v26; // rsi
  int v27; // eax
  int v28; // r9d

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == *a2 )
    {
LABEL_3:
      *v8 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v27 = 1;
      while ( v9 != -4096 )
      {
        v28 = v27 + 1;
        v7 = v6 & (v27 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( *a2 == *v8 )
          goto LABEL_3;
        v27 = v28;
      }
    }
  }
  v10 = *(_QWORD *)(a1 + 32);
  v11 = *(_DWORD *)(a1 + 40);
  v12 = 0x6DB6DB6DB6DB6DB7LL * ((v10 + 56LL * v11 - (__int64)(a2 + 7)) >> 3);
  if ( v10 + 56LL * v11 - (__int64)(a2 + 7) > 0 )
  {
    v13 = a2 + 2;
    do
    {
      v14 = v13[1];
      *(v13 - 2) = v13[5];
      while ( v14 )
      {
        sub_2754510(*(_QWORD *)(v14 + 24));
        v15 = v14;
        v14 = *(_QWORD *)(v14 + 16);
        j_j___libc_free_0(v15);
      }
      v16 = v13[8];
      v13[1] = 0;
      v13[2] = v13;
      v13[3] = v13;
      v13[4] = 0;
      if ( v16 )
      {
        v17 = *((_DWORD *)v13 + 14);
        v13[1] = v16;
        *(_DWORD *)v13 = v17;
        v13[2] = v13[9];
        v13[3] = v13[10];
        *(_QWORD *)(v16 + 8) = v13;
        v18 = v13[11];
        v13[8] = 0;
        v13[4] = v18;
        v19 = v13 + 7;
        v13[9] = v13 + 7;
        v13[10] = v13 + 7;
        v13[11] = 0;
      }
      else
      {
        v19 = v13 + 7;
      }
      v13 = v19;
      --v12;
    }
    while ( v12 );
    v11 = *(_DWORD *)(a1 + 40);
    v10 = *(_QWORD *)(a1 + 32);
  }
  v20 = v11 - 1;
  *(_DWORD *)(a1 + 40) = v20;
  sub_2754510(*(_QWORD *)(v10 + 56 * v20 + 24));
  v21 = *(_QWORD *)(a1 + 32);
  result = v21 + 56LL * *(unsigned int *)(a1 + 40);
  if ( a2 != (_QWORD *)result )
  {
    v23 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)a2 - v21) >> 3);
    result = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)result )
    {
      v24 = *(_QWORD **)(a1 + 8);
      v25 = &v24[2 * *(unsigned int *)(a1 + 24)];
      if ( v24 != v25 )
      {
        while ( 1 )
        {
          result = (__int64)v24;
          if ( *v24 != -4096 && *v24 != -8192 )
            break;
          v24 += 2;
          if ( v25 == v24 )
            return result;
        }
        if ( v25 != v24 )
        {
          do
          {
            v26 = *(unsigned int *)(result + 8);
            if ( v23 < v26 )
              *(_DWORD *)(result + 8) = v26 - 1;
            result += 16;
            if ( (_QWORD *)result == v25 )
              break;
            while ( *(_QWORD *)result == -4096 || *(_QWORD *)result == -8192 )
            {
              result += 16;
              if ( v25 == (_QWORD *)result )
                return result;
            }
          }
          while ( (_QWORD *)result != v25 );
        }
      }
    }
  }
  return result;
}

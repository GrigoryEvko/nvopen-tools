// Function: sub_2398F30
// Address: 0x2398f30
//
__int64 __fastcall sub_2398F30(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  _QWORD **v4; // rbx
  __int64 v5; // rdx
  _QWORD **v6; // r14
  _QWORD **v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rbx
  unsigned __int64 v10; // r15
  __int64 v11; // rdi
  _QWORD **i; // rbx
  __int64 v13; // rax
  _QWORD *v14; // r12
  unsigned __int64 v15; // r8
  __int64 v16; // rdi
  int v17; // edx
  int v18; // r14d
  unsigned int v19; // r15d
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  __int64 j; // rdx
  __int64 v24; // [rsp+0h] [rbp-40h]
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  result = (unsigned int)(4 * v2);
  v4 = *(_QWORD ***)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v24 = 32 * v5;
  v6 = &v4[4 * v5];
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    v7 = v4 + 1;
    if ( v6 != v4 )
    {
      while ( 1 )
      {
        v8 = (__int64)*(v7 - 1);
        if ( v8 != -4096 )
        {
          if ( v8 != -8192 )
          {
            v9 = *v7;
            while ( v7 != v9 )
            {
              v10 = (unsigned __int64)v9;
              v9 = (_QWORD *)*v9;
              v11 = *(_QWORD *)(v10 + 24);
              if ( v11 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
              j_j___libc_free_0(v10);
            }
          }
          *(v7 - 1) = (_QWORD *)-4096LL;
        }
        result = (__int64)(v7 + 4);
        if ( v6 == v7 + 3 )
          break;
        v7 += 4;
      }
    }
    goto LABEL_16;
  }
  for ( i = v4 + 1; ; i += 4 )
  {
    v13 = (__int64)*(i - 1);
    if ( v13 != -8192 && v13 != -4096 )
    {
      v14 = *i;
      while ( v14 != i )
      {
        v15 = (unsigned __int64)v14;
        v14 = (_QWORD *)*v14;
        v16 = *(_QWORD *)(v15 + 24);
        if ( v16 )
        {
          v25 = v15;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
          v15 = v25;
        }
        j_j___libc_free_0(v15);
      }
    }
    result = (__int64)(i + 4);
    if ( v6 == i + 3 )
      break;
  }
  v17 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( !v17 )
    {
LABEL_16:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v24, 8);
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_40;
  }
  v18 = 64;
  v19 = v2 - 1;
  if ( v19 )
  {
    _BitScanReverse(&v20, v19);
    v18 = 1 << (33 - (v20 ^ 0x1F));
    if ( v18 < 64 )
      v18 = 64;
  }
  v21 = *(_QWORD **)(a1 + 8);
  if ( v18 != v17 )
  {
    sub_C7D6A0((__int64)v21, v24, 8);
    result = sub_2309150(v18);
    *(_DWORD *)(a1 + 24) = result;
    if ( (_DWORD)result )
    {
      result = sub_C7D670(32LL * (unsigned int)result, 8);
      v22 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( j = result + 32 * v22; j != result; result += 32 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
      return result;
    }
LABEL_40:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v21[4 * (unsigned int)v18];
  do
  {
    if ( v21 )
      *v21 = -4096;
    v21 += 4;
  }
  while ( (_QWORD *)result != v21 );
  return result;
}

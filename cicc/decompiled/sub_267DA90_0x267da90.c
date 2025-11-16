// Function: sub_267DA90
// Address: 0x267da90
//
__int64 __fastcall sub_267DA90(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  _QWORD **v4; // r13
  _QWORD **j; // rbx
  __int64 v6; // rax
  _QWORD *v7; // r14
  unsigned __int64 v8; // r15
  __int64 v9; // rdi
  __int64 result; // rax
  unsigned int v11; // r14d
  __int64 v12; // rbx
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 i; // rdx
  __int64 v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+8h] [rbp-38h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v2 = *(unsigned int *)(a1 + 24);
  v20 = v2;
  v21 = *(_DWORD *)(a1 + 16);
  if ( !(_DWORD)v2 )
  {
    result = v21;
    if ( !v21 )
      goto LABEL_34;
    v11 = v21 - 1;
    if ( v21 == 1 )
    {
      v13 = *(_QWORD **)(a1 + 8);
      LODWORD(v12) = 64;
LABEL_21:
      sub_C7D6A0((__int64)v13, 32LL * v20, 8);
      v14 = 4 * v12;
      v15 = (((((((v14 / 3 + 1) | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 2)
              | (v14 / 3 + 1)
              | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 4)
            | (((v14 / 3 + 1) | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 2)
            | (v14 / 3 + 1)
            | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 8)
          | (((((v14 / 3 + 1) | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 2)
            | (v14 / 3 + 1)
            | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 4)
          | (((v14 / 3 + 1) | ((unsigned __int64)(v14 / 3 + 1) >> 1)) >> 2)
          | (v14 / 3 + 1)
          | ((unsigned __int64)(v14 / 3 + 1) >> 1);
      v16 = ((v15 >> 16) | v15) + 1;
      *(_DWORD *)(a1 + 24) = v16;
      result = sub_C7D670(32 * v16, 8);
      v17 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 32 * v17; i != result; result += 32 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
      return result;
    }
    LODWORD(result) = 0;
    goto LABEL_18;
  }
  v3 = *(_QWORD *)(a1 + 8);
  v19 = *(unsigned int *)(a1 + 24);
  v4 = (_QWORD **)(v3 + 32 * v2);
  for ( j = (_QWORD **)(v3 + 8); ; j += 4 )
  {
    v6 = (__int64)*(j - 1);
    if ( v6 != -4096 && v6 != -8192 )
    {
      v7 = *j;
      while ( v7 != j )
      {
        v8 = (unsigned __int64)v7;
        v7 = (_QWORD *)*v7;
        v9 = *(_QWORD *)(v8 + 24);
        if ( v9 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
        j_j___libc_free_0(v8);
      }
    }
    if ( v4 == j + 3 )
      break;
  }
  result = *(unsigned int *)(a1 + 24);
  if ( !v21 )
  {
    if ( (_DWORD)result )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 32 * v19, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
LABEL_34:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v12 = 64;
  v11 = v21 - 1;
  if ( v21 != 1 )
  {
LABEL_18:
    _BitScanReverse(&v11, v11);
    v12 = (unsigned int)(1 << (33 - (v11 ^ 0x1F)));
    if ( (int)v12 < 64 )
      v12 = 64;
  }
  v13 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v12 != (_DWORD)result )
    goto LABEL_21;
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v13[4 * v12];
  do
  {
    if ( v13 )
      *v13 = -4096;
    v13 += 4;
  }
  while ( (_QWORD *)result != v13 );
  return result;
}

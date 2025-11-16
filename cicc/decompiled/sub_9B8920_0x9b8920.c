// Function: sub_9B8920
// Address: 0x9b8920
//
__int64 __fastcall sub_9B8920(unsigned int a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 *v4; // r8
  unsigned int v7; // r13d
  int v8; // r12d
  int v9; // r15d
  bool v10; // cc
  signed int v11; // esi
  __int64 result; // rax
  unsigned int v13; // ecx
  __int64 v14; // rdx
  int v15; // edx
  int v16; // eax
  unsigned int v17; // eax
  __int64 *v18; // r8
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 *v22; // [rsp+8h] [rbp-48h]
  __int64 *v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-38h]

  v4 = a3;
  v26 = *((_DWORD *)a2 + 2);
  v7 = v26;
  v8 = (int)v26 / (int)(a1 >> 7);
  v9 = v8 / 2;
  if ( v26 > 0x40 )
  {
    sub_C43690(&v25, 0, 0);
    v18 = a3;
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    {
      j_j___libc_free_0_0(*a3);
      v18 = a3;
      *a3 = v25;
      v19 = v26;
      v26 = v7;
      *((_DWORD *)a3 + 2) = v19;
    }
    else
    {
      *a3 = v25;
      v20 = v26;
      v26 = v7;
      *((_DWORD *)a3 + 2) = v20;
    }
    v24 = v18;
    sub_C43690(&v25, 0, 0);
    v4 = v24;
  }
  else
  {
    v10 = *((_DWORD *)a3 + 2) <= 0x40u;
    v25 = 0;
    if ( v10 )
    {
      *a3 = 0;
      *((_DWORD *)a3 + 2) = v7;
    }
    else if ( *a3 )
    {
      j_j___libc_free_0_0(*a3);
      v4 = a3;
      *a3 = v25;
      v17 = v26;
      v26 = v7;
      *((_DWORD *)a3 + 2) = v17;
    }
    else
    {
      *((_DWORD *)a3 + 2) = v26;
    }
    v25 = 0;
  }
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
  {
    v22 = v4;
    j_j___libc_free_0_0(*a4);
    v4 = v22;
  }
  v11 = 0;
  *a4 = v25;
  result = v26;
  *((_DWORD *)a4 + 2) = v26;
  if ( v7 )
  {
    do
    {
      while ( 1 )
      {
        result = *a2;
        if ( *((_DWORD *)a2 + 2) > 0x40u )
          result = *(_QWORD *)(result + 8LL * ((unsigned int)v11 >> 6));
        if ( (result & (1LL << v11)) == 0 )
          goto LABEL_12;
        v15 = v11 % v8;
        v16 = v8 * (v11 / v8);
        if ( v9 <= v11 % v8 )
          break;
        v13 = v16 + 2 * v15;
        result = *v4;
        v14 = 1LL << v13;
        if ( *((_DWORD *)v4 + 2) > 0x40u )
          goto LABEL_22;
        result |= v14;
        *v4 = result;
LABEL_12:
        if ( v7 == ++v11 )
          return result;
      }
      v13 = v16 + 2 * (v15 - v9);
      result = *a4;
      v14 = 1LL << v13;
      if ( *((_DWORD *)a4 + 2) > 0x40u )
      {
LABEL_22:
        *(_QWORD *)(result + 8LL * (v13 >> 6)) |= v14;
        goto LABEL_12;
      }
      result |= v14;
      ++v11;
      *a4 = result;
    }
    while ( v7 != v11 );
  }
  return result;
}

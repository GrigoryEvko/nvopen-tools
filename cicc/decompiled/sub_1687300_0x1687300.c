// Function: sub_1687300
// Address: 0x1687300
//
_DWORD *__fastcall sub_1687300(_DWORD *a1, __int64 a2, __int64 a3)
{
  _DWORD *v3; // r12
  const void *v4; // r15
  _DWORD *v5; // rdx
  int v6; // eax
  int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // r13d
  _DWORD *result; // rax
  __int64 v13; // rdi
  _QWORD *v14; // rax
  int v15; // r9d
  size_t v16; // rdx
  size_t v17; // r8
  __int64 v18; // rcx
  _DWORD *v19; // r14
  _QWORD *v20; // rdi
  __int64 v21; // rdi
  int v22; // edx
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  char v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  size_t n; // [rsp+10h] [rbp-40h]
  size_t na; // [rsp+10h] [rbp-40h]

  if ( a1 )
  {
    v3 = a1;
    v4 = a1 + 1;
    if ( a1[1] == -1 )
    {
      v9 = 8;
      v10 = 4;
      LODWORD(v8) = 1;
    }
    else
    {
      v5 = a1 + 1;
      v6 = 0;
      do
      {
        ++v5;
        v7 = v6++;
      }
      while ( *v5 != -1 );
      v8 = (unsigned int)(v7 + 2);
      v9 = 4LL * (unsigned int)(v7 + 3);
      v10 = 4 * v8;
    }
    v11 = *a1;
    if ( *a1 < (unsigned int)v8 )
    {
      v27 = v9;
      v29 = v10;
      n = 4LL * (2 * v11 + 2);
      v13 = *(_QWORD *)(sub_1689050(a1, a2, n) + 24);
      v14 = sub_1685080(v13, n);
      v16 = n;
      v17 = v29;
      v18 = v27;
      v19 = v14;
      if ( !v14 )
      {
        sub_1683C30(v13, 4 * (2 * v11 + 2), n, v27, v29, v15, v27);
        v18 = v28;
        v17 = v29;
        v16 = 4LL * (2 * v11 + 2);
      }
      v30 = v18;
      na = v17;
      memset(v19, 0, v16);
      memcpy(v19 + 1, v4, 4LL * v11);
      v20 = v3;
      v3 = v19;
      *v19 = 2 * v11;
      sub_16856A0(v20);
      v9 = v30;
      v10 = na;
    }
    *(_DWORD *)((char *)v3 + v10) = a2;
    *(_DWORD *)((char *)v3 + v9) = -1;
    return v3;
  }
  else
  {
    v21 = *(_QWORD *)(sub_1689050(0, a2, a3) + 24);
    result = sub_1685080(v21, 12);
    if ( !result )
    {
      sub_1683C30(v21, 12, v22, v23, v24, v25, v26);
      result = 0;
    }
    *result = 1;
    result[1] = a2;
    result[2] = -1;
  }
  return result;
}

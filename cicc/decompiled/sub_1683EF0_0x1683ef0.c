// Function: sub_1683EF0
// Address: 0x1683ef0
//
_DWORD *__fastcall sub_1683EF0(_DWORD *a1, int a2)
{
  _DWORD *v2; // r12
  const void *v3; // r15
  _DWORD *v4; // rdx
  int v5; // eax
  int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // r13d
  _DWORD *result; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  int v14; // r9d
  size_t v15; // rdx
  size_t v16; // r8
  __int64 v17; // rcx
  _DWORD *v18; // r14
  _DWORD *v19; // rdi
  __int64 v20; // rdi
  int v21; // edx
  int v22; // ecx
  int v23; // r8d
  int v24; // r9d
  char v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  size_t n; // [rsp+10h] [rbp-40h]
  size_t na; // [rsp+10h] [rbp-40h]

  if ( a1 )
  {
    v2 = a1;
    v3 = a1 + 1;
    if ( a1[1] == -1 )
    {
      v8 = 8;
      v9 = 4;
      LODWORD(v7) = 1;
    }
    else
    {
      v4 = a1 + 1;
      v5 = 0;
      do
      {
        ++v4;
        v6 = v5++;
      }
      while ( *v4 != -1 );
      v7 = (unsigned int)(v6 + 2);
      v8 = 4LL * (unsigned int)(v6 + 3);
      v9 = 4 * v7;
    }
    v10 = *a1;
    if ( *a1 < (unsigned int)v7 )
    {
      v26 = v8;
      v28 = v9;
      n = 4LL * (2 * v10 + 2);
      v12 = *(_QWORD *)(sub_1689050() + 24);
      v13 = sub_1685080(v12, n);
      v15 = n;
      v16 = v28;
      v17 = v26;
      v18 = (_DWORD *)v13;
      if ( !v13 )
      {
        sub_1683C30(v12, n, n, v26, v28, v14, v26);
        v17 = v27;
        v16 = v28;
        v15 = 4LL * (2 * v10 + 2);
      }
      v29 = v17;
      na = v16;
      memset(v18, 0, v15);
      memcpy(v18 + 1, v3, 4LL * v10);
      v19 = v2;
      v2 = v18;
      *v18 = 2 * v10;
      sub_16856A0(v19);
      v8 = v29;
      v9 = na;
    }
    *(_DWORD *)((char *)v2 + v9) = a2;
    *(_DWORD *)((char *)v2 + v8) = -1;
    return v2;
  }
  else
  {
    v20 = *(_QWORD *)(sub_1689050() + 24);
    result = (_DWORD *)sub_1685080(v20, 12);
    if ( !result )
    {
      sub_1683C30(v20, 12, v21, v22, v23, v24, v25);
      result = 0;
    }
    *result = 1;
    result[1] = a2;
    result[2] = -1;
  }
  return result;
}

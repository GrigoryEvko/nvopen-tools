// Function: sub_254C7C0
// Address: 0x254c7c0
//
char __fastcall sub_254C7C0(__int64 *a1, __int64 a2)
{
  char result; // al
  unsigned int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // [rsp-58h] [rbp-58h] BYREF
  __int64 *v14; // [rsp-50h] [rbp-50h]
  __int64 *v15; // [rsp-48h] [rbp-48h]
  __int64 v16; // [rsp-40h] [rbp-40h]
  _QWORD v17[7]; // [rsp-38h] [rbp-38h] BYREF

  if ( a1 == (__int64 *)a2 )
    return 1;
  result = a1 + 1024 == 0 || a1 + 512 == 0 || a2 == -8192 || a2 == -4096;
  if ( result )
    return 0;
  if ( a1 )
  {
    v3 = *((_DWORD *)a1 + 5) - *((_DWORD *)a1 + 6);
    if ( !a2 )
      return *((_DWORD *)a1 + 5) == *((_DWORD *)a1 + 6);
  }
  else
  {
    v3 = 0;
    if ( !a2 )
      return 1;
  }
  if ( *(_DWORD *)(a2 + 20) - *(_DWORD *)(a2 + 24) != v3 )
    return result;
  if ( !v3 )
    return 1;
  if ( v3 < *((_DWORD *)a1 + 5) - *((_DWORD *)a1 + 6) )
    return result;
  v4 = a1[1];
  v5 = *((_BYTE *)a1 + 28) ? v4 + 8LL * *((unsigned int *)a1 + 5) : v4 + 8LL * *((unsigned int *)a1 + 4);
  v13 = (__int64 *)a1[1];
  v14 = (__int64 *)v5;
  sub_254BBF0((__int64)&v13);
  v6 = *a1;
  v7 = *((_BYTE *)a1 + 28) == 0;
  v15 = a1;
  v16 = v6;
  v8 = v7 ? *((unsigned int *)a1 + 4) : *((unsigned int *)a1 + 5);
  v17[0] = a1[1] + 8 * v8;
  v17[1] = v17[0];
  sub_254BBF0((__int64)v17);
  v11 = *a1;
  v17[2] = a1;
  v17[3] = v11;
  v12 = v13;
  if ( (__int64 *)v17[0] == v13 )
    return 1;
  while ( 1 )
  {
    result = sub_B19060(a2, *v12, v9, v10);
    if ( !result )
      return result;
    v10 = (__int64)v14;
    v12 = v13 + 1;
    v13 = v12;
    if ( v12 == v14 )
    {
LABEL_19:
      if ( v12 == (__int64 *)v17[0] )
        return 1;
    }
    else
    {
      while ( 1 )
      {
        v9 = *v12 + 2;
        if ( v9 > 1 )
          break;
        v13 = ++v12;
        if ( v14 == v12 )
          goto LABEL_19;
      }
      v12 = v13;
      if ( v13 == (__int64 *)v17[0] )
        return 1;
    }
  }
}

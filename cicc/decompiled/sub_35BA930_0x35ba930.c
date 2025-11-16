// Function: sub_35BA930
// Address: 0x35ba930
//
_DWORD *__fastcall sub_35BA930(_QWORD *a1, unsigned int a2, __int64 a3)
{
  int v3; // r9d
  __int64 v5; // r8
  _DWORD *result; // rax
  __int64 v7; // rax
  int v8; // eax
  int v9; // eax
  _DWORD *v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // [rsp+8h] [rbp-18h] BYREF
  int v13[5]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = a2;
  v5 = *(_QWORD *)(*a1 + 160LL) + 96LL * a2;
  result = (_DWORD *)(*(_QWORD *)(v5 + 80) - *(_QWORD *)(v5 + 72));
  if ( result == (_DWORD *)12 )
  {
    v12 = a2;
    v13[0] = a2;
    v9 = *(_DWORD *)(v5 + 16);
    switch ( v9 )
    {
      case 2:
        sub_35B9090(a1 + 7, (unsigned int *)v13);
        break;
      case 3:
        sub_35B9090(a1 + 1, (unsigned int *)v13);
        break;
      case 1:
        sub_35B9090(a1 + 13, (unsigned int *)v13);
        break;
    }
    sub_B99820((__int64)(a1 + 1), &v12);
    result = (_DWORD *)(*(_QWORD *)(*a1 + 160LL) + 96LL * v12);
    result[4] = 3;
  }
  else if ( *(_DWORD *)a3 == 1 )
  {
    v7 = *(unsigned int *)(a3 + 4);
    if ( *(_DWORD *)(a3 + 8) < (unsigned int)v7
      || (v10 = *(_DWORD **)(a3 + 16),
          v13[0] = 0,
          v11 = (__int64)&v10[v7],
          result = sub_35B8490(v10, v11, v13),
          (_DWORD *)v11 != result) )
    {
      v12 = v3;
      v13[0] = v3;
      v8 = *(_DWORD *)(v5 + 16);
      switch ( v8 )
      {
        case 2:
          sub_35B9090(a1 + 7, (unsigned int *)v13);
          break;
        case 3:
          sub_35B9090(a1 + 1, (unsigned int *)v13);
          break;
        case 1:
          sub_35B9090(a1 + 13, (unsigned int *)v13);
          break;
      }
      sub_B99820((__int64)(a1 + 7), &v12);
      result = (_DWORD *)(*(_QWORD *)(*a1 + 160LL) + 96LL * v12);
      result[4] = 2;
      *((_BYTE *)result + 64) = 1;
    }
  }
  return result;
}

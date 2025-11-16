// Function: sub_1ECCA20
// Address: 0x1ecca20
//
_DWORD *__fastcall sub_1ECCA20(__int64 *a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r14
  _DWORD *result; // rax
  __int64 v7; // rax
  _DWORD *v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // r15
  __int64 *v11; // r12
  _BOOL4 v12; // r8d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rax
  _BOOL4 v16; // [rsp+Ch] [rbp-44h]
  int v17[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = 88LL * a2;
  result = (_DWORD *)(*(_QWORD *)(v4 + *(_QWORD *)(*a1 + 160) + 72) - *(_QWORD *)(v4 + *(_QWORD *)(*a1 + 160) + 64));
  if ( result == (_DWORD *)12 )
  {
    v10 = a1 + 2;
    sub_1ECBF60(a1, a2);
    v11 = (__int64 *)a1[3];
    if ( v11 )
    {
      while ( 1 )
      {
        v13 = *((_DWORD *)v11 + 8);
        v14 = (__int64 *)v11[3];
        if ( a2 < v13 )
          v14 = (__int64 *)v11[2];
        if ( !v14 )
          break;
        v11 = v14;
      }
      if ( a2 >= v13 )
      {
        if ( a2 <= v13 )
        {
LABEL_19:
          result = *(_DWORD **)(*a1 + 160);
          result[(unsigned __int64)v4 / 4 + 4] = 3;
          return result;
        }
        goto LABEL_16;
      }
      if ( v11 == (__int64 *)a1[4] )
      {
LABEL_16:
        v12 = 1;
        if ( v10 != v11 )
          v12 = a2 < *((_DWORD *)v11 + 8);
        goto LABEL_18;
      }
    }
    else
    {
      v11 = a1 + 2;
      if ( v10 == (__int64 *)a1[4] )
      {
        v12 = 1;
LABEL_18:
        v16 = v12;
        v15 = sub_22077B0(40);
        *(_DWORD *)(v15 + 32) = a2;
        sub_220F040(v16, v15, v11, a1 + 2);
        ++a1[6];
        goto LABEL_19;
      }
    }
    if ( a2 <= *(_DWORD *)(sub_220EF80(v11) + 32) || !v11 )
      goto LABEL_19;
    goto LABEL_16;
  }
  if ( *(_DWORD *)a3 == 1 )
  {
    v7 = *(unsigned int *)(a3 + 4);
    if ( *(_DWORD *)(a3 + 8) < (unsigned int)v7
      || (v8 = *(_DWORD **)(a3 + 16),
          v17[0] = 0,
          v9 = (__int64)&v8[v7],
          result = sub_1ECB090(v8, v9, v17),
          (_DWORD *)v9 != result) )
    {
      v17[0] = a2;
      sub_1ECBF60(a1, a2);
      sub_1DF87A0(a1 + 7, (unsigned int *)v17);
      result = (_DWORD *)(*(_QWORD *)(*a1 + 160) + 88LL * (unsigned int)v17[0]);
      result[4] = 2;
    }
  }
  return result;
}

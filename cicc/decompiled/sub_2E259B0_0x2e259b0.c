// Function: sub_2E259B0
// Address: 0x2e259b0
//
__int64 __fastcall sub_2E259B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v9; // rsi
  unsigned int v10; // r15d
  __int64 result; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rsi
  unsigned int v14; // ecx
  unsigned int v15; // edx
  __int64 v16; // r9
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r12

  v9 = (_BYTE *)a2[4];
  v10 = *(_DWORD *)(a4 + 24);
  result = (__int64)(a2[5] - (_QWORD)v9) >> 3;
  if ( (_DWORD)result )
  {
    v12 = (__int64)&v9[8 * (unsigned int)(result - 1) + 8];
    while ( 1 )
    {
      result = *(_QWORD *)v9;
      if ( a4 == *(_QWORD *)(*(_QWORD *)v9 + 24LL) )
        break;
      v9 += 8;
      if ( v9 == (_BYTE *)v12 )
        goto LABEL_6;
    }
    result = (__int64)sub_2E25970((__int64)(a2 + 4), v9);
  }
LABEL_6:
  if ( a4 != a3 )
  {
    v13 = (_QWORD *)*a2;
    if ( a2 == (_QWORD *)*a2 )
      goto LABEL_19;
    result = a2[3];
    if ( a2 == (_QWORD *)result )
    {
      result = a2[1];
      v15 = v10 >> 7;
      a2[3] = result;
      v14 = *(_DWORD *)(result + 16);
      if ( v10 >> 7 == v14 )
        goto LABEL_17;
    }
    else
    {
      v14 = *(_DWORD *)(result + 16);
      v15 = v10 >> 7;
      if ( v14 == v10 >> 7 )
      {
LABEL_18:
        if ( v15 == *(_DWORD *)(result + 16) && (*(_QWORD *)(result + 8LL * ((v10 >> 6) & 1) + 24) & (1LL << v10)) != 0 )
          return result;
        goto LABEL_19;
      }
    }
    if ( v14 > v15 )
    {
      if ( v13 != (_QWORD *)result )
      {
        do
          result = *(_QWORD *)(result + 8);
        while ( v13 != (_QWORD *)result && *(_DWORD *)(result + 16) > v15 );
      }
    }
    else if ( a2 != (_QWORD *)result )
    {
      while ( v14 < v15 )
      {
        result = *(_QWORD *)result;
        if ( a2 == (_QWORD *)result )
          break;
        v14 = *(_DWORD *)(result + 16);
      }
    }
    a2[3] = result;
LABEL_17:
    if ( a2 != (_QWORD *)result )
      goto LABEL_18;
LABEL_19:
    sub_FDE240(a2, v10);
    v17 = *(_QWORD *)(a4 + 64);
    v18 = *(unsigned int *)(a5 + 8);
    v19 = *(unsigned int *)(a4 + 72);
    v20 = v19;
    if ( v18 + v19 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v18 + v19, 8u, v18 + v19, v16);
      v18 = *(unsigned int *)(a5 + 8);
    }
    result = *(_QWORD *)a5 + 8 * v18;
    if ( 8 * v19 )
    {
      do
      {
        result += 8;
        *(_QWORD *)(result - 8) = *(_QWORD *)(v17 + 8 * v20-- - 8);
      }
      while ( v20 );
      LODWORD(v18) = *(_DWORD *)(a5 + 8);
    }
    *(_DWORD *)(a5 + 8) = v18 + v19;
  }
  return result;
}

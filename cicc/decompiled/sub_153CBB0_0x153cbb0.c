// Function: sub_153CBB0
// Address: 0x153cbb0
//
__int64 __fastcall sub_153CBB0(__int64 **a1, __int64 **a2, __int64 **a3, __int64 **a4, __int64 a5, __int64 a6)
{
  __int64 **v9; // rbx
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rdi
  __int64 result; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // rsi
  __int64 v20; // rax
  _QWORD v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v21[0] = a6;
  if ( a2 == a1 )
  {
LABEL_7:
    v12 = (char *)a4 - (char *)a3;
    v13 = v12 >> 4;
    if ( v12 <= 0 )
      return a5;
    goto LABEL_8;
  }
  v9 = a1;
  while ( a4 != a3 )
  {
    if ( sub_153CA80((__int64)v21, a3, v9) )
    {
      v10 = *a3;
      a5 += 16;
      a3 += 2;
      *(_QWORD *)(a5 - 16) = v10;
      *(_DWORD *)(a5 - 8) = *((_DWORD *)a3 - 2);
      if ( v9 == a2 )
        goto LABEL_7;
    }
    else
    {
      v11 = *v9;
      a5 += 16;
      v9 += 2;
      *(_QWORD *)(a5 - 16) = v11;
      *(_DWORD *)(a5 - 8) = *((_DWORD *)v9 - 2);
      if ( v9 == a2 )
        goto LABEL_7;
    }
  }
  result = a5;
  v17 = (char *)a2 - (char *)v9;
  v18 = ((char *)a2 - (char *)v9) >> 4;
  if ( (char *)a2 - (char *)v9 > 0 )
  {
    do
    {
      v19 = *v9;
      result += 16;
      v9 += 2;
      *(_QWORD *)(result - 16) = v19;
      *(_DWORD *)(result - 8) = *((_DWORD *)v9 - 2);
      --v18;
    }
    while ( v18 );
    v20 = 16;
    if ( v17 > 0 )
      v20 = v17;
    v12 = (char *)a4 - (char *)a3;
    a5 += v20;
    v13 = v12 >> 4;
    if ( v12 <= 0 )
      return a5;
LABEL_8:
    v14 = a5;
    do
    {
      v15 = *a3;
      v14 += 16;
      a3 += 2;
      *(_QWORD *)(v14 - 16) = v15;
      *(_DWORD *)(v14 - 8) = *((_DWORD *)a3 - 2);
      --v13;
    }
    while ( v13 );
    return a5 + v12;
  }
  return result;
}

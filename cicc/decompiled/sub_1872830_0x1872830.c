// Function: sub_1872830
// Address: 0x1872830
//
__int64 __fastcall sub_1872830(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 i; // rcx
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rcx

  v7 = a3 & 1;
  v8 = (a3 - 1) / 2;
  if ( a2 >= v8 )
  {
    result = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v10 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v10 )
  {
    v10 = 2 * (i + 1);
    result = a1 + 32 * (i + 1);
    if ( *(_DWORD *)(result + 8) < *(_DWORD *)(a1 + 16 * (v10 - 1) + 8) )
      result = a1 + 16 * --v10;
    v12 = a1 + 16 * i;
    *(_QWORD *)v12 = *(_QWORD *)result;
    *(_DWORD *)(v12 + 8) = *(_DWORD *)(result + 8);
    if ( v10 >= v8 )
      break;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v10 )
    {
      v10 = 2 * v10 + 1;
      v15 = a1 + 16 * v10;
      *(_QWORD *)result = *(_QWORD *)v15;
      *(_DWORD *)(result + 8) = *(_DWORD *)(v15 + 8);
      result = v15;
    }
  }
  v13 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      result = a1 + 16 * v10;
      v14 = a1 + 16 * v13;
      if ( a5 <= *(_DWORD *)(v14 + 8) )
        break;
      *(_QWORD *)result = *(_QWORD *)v14;
      *(_DWORD *)(result + 8) = *(_DWORD *)(v14 + 8);
      v10 = v13;
      if ( a2 >= v13 )
      {
        *(_QWORD *)v14 = a4;
        *(_DWORD *)(v14 + 8) = a5;
        return a1 + 16 * v13;
      }
      v13 = (v13 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)result = a4;
  *(_DWORD *)(result + 8) = a5;
  return result;
}

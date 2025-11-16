// Function: sub_1057F60
// Address: 0x1057f60
//
__int64 __fastcall sub_1057F60(__int64 a1, unsigned __int8 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 result; // rax
  unsigned int v9; // edx
  _BYTE *v10; // rsi
  unsigned __int8 *v11; // [rsp+8h] [rbp-18h] BYREF

  if ( (unsigned int)*a2 - 30 > 0xA )
  {
    result = sub_1057F50(a1, (__int64)a2);
LABEL_9:
    if ( !(_BYTE)result )
      return result;
    goto LABEL_10;
  }
  v7 = *((_QWORD *)a2 + 5);
  if ( !*(_BYTE *)(a1 + 300) )
  {
LABEL_8:
    sub_C8CC70(a1 + 272, v7, (__int64)a3, a4, a5, a6);
    result = v9;
    goto LABEL_9;
  }
  result = *(_QWORD *)(a1 + 280);
  a4 = *(unsigned int *)(a1 + 292);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_15:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 288) )
    {
      *(_DWORD *)(a1 + 292) = a4 + 1;
      *a3 = v7;
      ++*(_QWORD *)(a1 + 272);
LABEL_10:
      v11 = a2;
      v10 = *(_BYTE **)(a1 + 568);
      if ( v10 == *(_BYTE **)(a1 + 576) )
        return (__int64)sub_D79240(a1 + 560, v10, &v11);
      if ( v10 )
      {
        *(_QWORD *)v10 = a2;
        v10 = *(_BYTE **)(a1 + 568);
      }
      *(_QWORD *)(a1 + 568) = v10 + 8;
      return result;
    }
    goto LABEL_8;
  }
  while ( v7 != *(_QWORD *)result )
  {
    result += 8;
    if ( a3 == (__int64 *)result )
      goto LABEL_15;
  }
  return result;
}

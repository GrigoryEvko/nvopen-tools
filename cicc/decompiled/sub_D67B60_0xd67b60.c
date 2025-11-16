// Function: sub_D67B60
// Address: 0xd67b60
//
__int64 __fastcall sub_D67B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rsi

  v6 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      result = a1 + 16 * v6;
      if ( (unsigned int)a5 <= *(_DWORD *)(result + 8)
        && ((_DWORD)a5 != *(_DWORD *)(result + 8) || HIDWORD(a5) <= *(_DWORD *)(result + 12)) )
      {
        break;
      }
      v8 = a1 + 16 * a2;
      *(_QWORD *)v8 = *(_QWORD *)result;
      *(_DWORD *)(v8 + 8) = *(_DWORD *)(result + 8);
      *(_DWORD *)(v8 + 12) = *(_DWORD *)(result + 12);
      a2 = v6;
      if ( a3 >= v6 )
        goto LABEL_8;
      v6 = (v6 - 1) / 2;
    }
  }
  result = a1 + 16 * a2;
LABEL_8:
  *(_QWORD *)result = a4;
  *(_QWORD *)(result + 8) = a5;
  return result;
}

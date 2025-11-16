// Function: sub_1441B50
// Address: 0x1441b50
//
__int64 __fastcall sub_1441B50(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 )
  {
    if ( (unsigned __int8)sub_1441AE0((_QWORD *)a2) && **(_DWORD **)(a2 + 8) == 1 )
    {
      if ( (unsigned __int8)sub_1625980(a3, v8) )
      {
        v7 = v8[0];
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v7;
        return a1;
      }
    }
    else if ( a4 )
    {
      sub_1368C40(a1, a4, *(_QWORD *)(a3 + 40));
      return a1;
    }
  }
  *(_BYTE *)(a1 + 8) = 0;
  return a1;
}

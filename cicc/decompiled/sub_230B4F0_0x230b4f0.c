// Function: sub_230B4F0
// Address: 0x230b4f0
//
__int64 __fastcall sub_230B4F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned int v10; // [rsp+Ch] [rbp-34h]

  result = *(_QWORD *)(a1 + 8);
  v10 = (*(_QWORD *)(a1 + 16) - result) >> 3;
  if ( v10 )
  {
    v7 = 0;
    while ( 1 )
    {
      v8 = v7++;
      result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(result + 8 * v8) + 24LL))(
                 *(_QWORD *)(result + 8 * v8),
                 a2,
                 a3,
                 a4);
      if ( v10 <= v7 )
      {
        if ( v10 == v7 )
          return result;
      }
      else
      {
        v9 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v9 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 44);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v9 + 1;
          *v9 = 44;
        }
      }
      result = *(_QWORD *)(a1 + 8);
    }
  }
  return result;
}

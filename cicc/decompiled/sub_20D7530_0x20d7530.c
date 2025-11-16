// Function: sub_20D7530
// Address: 0x20d7530
//
__int64 __fastcall sub_20D7530(__int64 *a1, int a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v9; // rbx
  bool v11; // cl
  __int64 v12; // rdi
  bool v13; // [rsp+7h] [rbp-39h]

  result = a1[1];
  v7 = *a1;
  if ( a2 == *(_DWORD *)(result - 16) )
  {
    v9 = result - 16;
    v11 = a3 != 0;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 8);
      if ( v12 != a4 && v11 )
      {
        v13 = v11;
        sub_20D69B0(v12, a3, a1[18]);
        v11 = v13;
      }
      if ( v9 == v7 )
        break;
      if ( a2 != *(_DWORD *)(v9 - 16) )
      {
        result = a1[1];
        goto LABEL_11;
      }
      v9 -= 16;
    }
    result = a1[1];
    if ( a2 != *(_DWORD *)v9 )
      v9 += 16;
  }
  else
  {
    v9 = a1[1];
  }
LABEL_11:
  if ( v9 != result )
    a1[1] = v9;
  return result;
}

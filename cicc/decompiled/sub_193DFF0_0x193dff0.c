// Function: sub_193DFF0
// Address: 0x193dff0
//
unsigned __int64 __fastcall sub_193DFF0(__int64 a1, __int64 a2)
{
  int v2; // r14d
  unsigned __int64 result; // rax
  __int64 v4; // r15
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 v9; // rsi
  char v10; // r12
  int v11; // [rsp+Ch] [rbp-44h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  __int64 v13; // [rsp+18h] [rbp-38h]

  v2 = *(unsigned __int8 *)(a2 + 16);
  result = (unsigned int)(v2 - 61);
  if ( (unsigned __int8)(v2 - 61) <= 1u )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v13 = *(_QWORD *)(a1 + 24);
    v12 = *(_QWORD *)a2;
    v5 = sub_1456C90(v4, *(_QWORD *)a2);
    v6 = sub_15F2050(a2);
    v7 = sub_1632FA0(v6);
    result = *(_QWORD *)(v7 + 24);
    v8 = result + *(unsigned int *)(v7 + 32);
    if ( result != v8 )
    {
      while ( v5 != *(unsigned __int8 *)result )
      {
        if ( v8 == ++result )
          return result;
      }
      result = sub_1456C90(v4, **(_QWORD **)(a1 + 40));
      if ( v5 > result )
      {
        if ( !v13 || (v11 = sub_14A3350(v13), result = sub_14A3350(v13), v11 <= (int)result) )
        {
          v9 = *(_QWORD *)(a1 + 48);
          v10 = (_BYTE)v2 == 62;
          if ( v9 )
          {
            if ( v10 == *(_BYTE *)(a1 + 56) )
            {
              result = sub_1456C90(v4, v9);
              if ( v5 > result )
              {
                result = sub_1456E10(v4, v12);
                *(_QWORD *)(a1 + 48) = result;
              }
            }
          }
          else
          {
            result = sub_1456E10(v4, v12);
            *(_BYTE *)(a1 + 56) = v10;
            *(_QWORD *)(a1 + 48) = result;
          }
        }
      }
    }
  }
  return result;
}

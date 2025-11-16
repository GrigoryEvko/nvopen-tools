// Function: sub_22DC9A0
// Address: 0x22dc9a0
//
__int64 __fastcall sub_22DC9A0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // r15
  int v7; // r12d
  unsigned int v8; // r13d
  unsigned __int64 v9; // rsi
  _QWORD *v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v14; // [rsp+10h] [rbp-40h]
  unsigned __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1[4];
  v15[0] = a2;
  v14 = v4;
  sub_22DC880(a3, v15);
  sub_22DB4B0(a1, v15[0]);
  result = *(_QWORD *)(v15[0] + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result != v15[0] + 48 )
  {
    if ( !result )
      BUG();
    v6 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v6);
      v7 = result;
      if ( (_DWORD)result )
      {
        v8 = 0;
        do
        {
          result = sub_B46EC0(v6, v8);
          v9 = result;
          if ( v14 != result )
          {
            result = a3[2];
            if ( !result )
              goto LABEL_14;
            v10 = a3 + 1;
            do
            {
              while ( 1 )
              {
                v11 = *(_QWORD *)(result + 16);
                v12 = *(_QWORD *)(result + 24);
                if ( *(_QWORD *)(result + 32) >= v9 )
                  break;
                result = *(_QWORD *)(result + 24);
                if ( !v12 )
                  goto LABEL_12;
              }
              v10 = (_QWORD *)result;
              result = *(_QWORD *)(result + 16);
            }
            while ( v11 );
LABEL_12:
            if ( a3 + 1 == v10 || v10[4] > v9 )
LABEL_14:
              result = sub_22DC9A0(a1, v9, a3);
          }
          ++v8;
        }
        while ( v7 != v8 );
      }
    }
  }
  return result;
}

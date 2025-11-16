// Function: sub_1442900
// Address: 0x1442900
//
__int64 __fastcall sub_1442900(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 *v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]

  result = *(_QWORD *)(a1 + 192);
  v3 = *(_QWORD *)(result + 32);
  if ( !v3 )
  {
    v4 = *(unsigned int *)(a1 + 208);
    if ( (_DWORD)v4 )
    {
      result = *(unsigned int *)(a2 + 8);
      v6 = a2 + 16;
      v7 = 48 * v4;
      do
      {
        v8 = (__int64 *)(v3 + *(_QWORD *)(a1 + 200));
        v9 = v8[1];
        v10 = *v8;
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
        {
          v12 = v7;
          sub_16CD150(a2, v6, 0, 16);
          result = *(unsigned int *)(a2 + 8);
          v7 = v12;
        }
        v11 = (_QWORD *)(*(_QWORD *)a2 + 16 * result);
        v3 += 48;
        *v11 = v10;
        v11[1] = v9;
        result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = result;
      }
      while ( v7 != v3 );
    }
  }
  return result;
}

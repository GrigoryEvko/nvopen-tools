// Function: sub_3985430
// Address: 0x3985430
//
__int64 __fastcall sub_3985430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  const void *v10; // rsi
  __int64 v11; // r8
  __int64 *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // [rsp+0h] [rbp-40h]

  result = *(_QWORD *)(a1 + 192);
  v7 = *(_QWORD *)(result + 32);
  if ( !v7 )
  {
    v8 = *(unsigned int *)(a1 + 208);
    if ( (_DWORD)v8 )
    {
      result = *(unsigned int *)(a2 + 8);
      v10 = (const void *)(a2 + 16);
      v11 = 48 * v8;
      do
      {
        v12 = (__int64 *)(v7 + *(_QWORD *)(a1 + 200));
        v13 = v12[1];
        v14 = *v12;
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
        {
          v16 = v11;
          sub_16CD150(a2, v10, 0, 16, v11, a6);
          result = *(unsigned int *)(a2 + 8);
          v11 = v16;
        }
        v15 = (_QWORD *)(*(_QWORD *)a2 + 16 * result);
        v7 += 48;
        *v15 = v14;
        v15[1] = v13;
        result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = result;
      }
      while ( v11 != v7 );
    }
  }
  return result;
}

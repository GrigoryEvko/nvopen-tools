// Function: sub_1D2DE40
// Address: 0x1d2de40
//
__int64 *__fastcall sub_1D2DE40(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  int v4; // r14d
  __int64 v6; // rax
  __int64 *result; // rax
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 *v12; // [rsp+18h] [rbp-28h] BYREF

  v4 = a4;
  v6 = a2[4];
  if ( *(_QWORD *)v6 != a3 || *(_DWORD *)(v6 + 8) != (_DWORD)a4 )
  {
    v12 = 0;
    result = sub_1D19350((__int64)a1, (__int64)a2, a3, a4, (__int64 *)&v12);
    if ( result )
      return result;
    if ( v12 && !(unsigned __int8)sub_1D2D480((__int64)a1, (__int64)a2, v8) )
      v12 = 0;
    v9 = a2[4];
    if ( *(_QWORD *)v9 )
    {
      v10 = *(_QWORD *)(v9 + 32);
      **(_QWORD **)(v9 + 24) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v9 + 24);
    }
    *(_QWORD *)v9 = a3;
    *(_DWORD *)(v9 + 8) = v4;
    if ( a3 )
    {
      v11 = *(_QWORD *)(a3 + 48);
      *(_QWORD *)(v9 + 32) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 24) = v9 + 32;
      *(_QWORD *)(v9 + 24) = a3 + 48;
      *(_QWORD *)(a3 + 48) = v9;
    }
    sub_1D18440(a1, (__int64)a2);
    if ( v12 )
      sub_16BDA20(a1 + 40, a2, v12);
  }
  return a2;
}

// Function: sub_1D2E160
// Address: 0x1d2e160
//
__int64 *__fastcall sub_1D2E160(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *result; // rax
  unsigned int v11; // edx
  __int64 *v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // [rsp+8h] [rbp-28h] BYREF

  v7 = (__int64 *)a3;
  v8 = a3 + 16 * a4;
  v9 = a2[4];
  if ( v8 == a3 )
    return a2;
  while ( *(_QWORD *)a3 == *(_QWORD *)v9 && *(_DWORD *)(a3 + 8) == *(_DWORD *)(v9 + 8) )
  {
    a3 += 16;
    v9 += 40;
    if ( v8 == a3 )
      return a2;
  }
  v19 = 0;
  result = sub_1D19610((__int64)a1, (__int64)a2, v7, a4, (__int64 *)&v19);
  if ( !result )
  {
    if ( v19 && !(unsigned __int8)sub_1D2D480((__int64)a1, (__int64)a2, v11) )
      v19 = 0;
    if ( (_DWORD)a4 )
    {
      v12 = v7;
      v13 = 0;
      v14 = (__int64)&v7[2 * (unsigned int)(a4 - 1) + 2];
      do
      {
        while ( 1 )
        {
          v18 = v13 + a2[4];
          if ( *(_QWORD *)v18 != *v12 || *(_DWORD *)(v18 + 8) != *((_DWORD *)v12 + 2) )
            break;
          v12 += 2;
          v13 += 40;
          if ( v12 == (__int64 *)v14 )
            goto LABEL_20;
        }
        if ( *(_QWORD *)v18 )
        {
          v15 = *(_QWORD *)(v18 + 32);
          **(_QWORD **)(v18 + 24) = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 24) = *(_QWORD *)(v18 + 24);
        }
        *(_QWORD *)v18 = *v12;
        *(_DWORD *)(v18 + 8) = *((_DWORD *)v12 + 2);
        v16 = *v12;
        if ( *v12 )
        {
          v17 = *(_QWORD *)(v16 + 48);
          *(_QWORD *)(v18 + 32) = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 24) = v18 + 32;
          *(_QWORD *)(v18 + 24) = v16 + 48;
          *(_QWORD *)(v16 + 48) = v18;
        }
        v12 += 2;
        v13 += 40;
      }
      while ( v12 != (__int64 *)v14 );
    }
LABEL_20:
    sub_1D18440(a1, (__int64)a2);
    if ( v19 )
      sub_16BDA20(a1 + 40, a2, v19);
    return a2;
  }
  return result;
}

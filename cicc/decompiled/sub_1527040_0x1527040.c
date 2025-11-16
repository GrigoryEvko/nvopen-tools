// Function: sub_1527040
// Address: 0x1527040
//
__int64 __fastcall sub_1527040(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v5; // rsi
  _DWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v15; // rsi

  if ( a2 != *(_DWORD *)(a1 + 20) )
  {
    sub_1524D80((_DWORD *)a1, 3u, *(_DWORD *)(a1 + 16));
    sub_1524E40((_DWORD *)a1, 1u, 6);
    sub_1524E40((_DWORD *)a1, 1u, 6);
    sub_1525280((_DWORD *)a1, a2, 6);
    *(_DWORD *)(a1 + 20) = a2;
  }
  sub_15253D0((_DWORD *)a1, *a3);
  v5 = *(_QWORD *)(a1 + 80);
  v6 = *(_DWORD **)(a1 + 72);
  if ( (_DWORD *)v5 == v6 || (v7 = v5 - 32, a2 != *(_DWORD *)(v5 - 32)) )
  {
    v8 = (v5 - (__int64)v6) >> 5;
    if ( !(_DWORD)v8 )
    {
LABEL_14:
      if ( v5 == *(_QWORD *)(a1 + 88) )
      {
        sub_1526090((__int64 *)(a1 + 72), (char *)v5);
        v15 = *(_QWORD *)(a1 + 80);
      }
      else
      {
        if ( v5 )
        {
          *(_DWORD *)v5 = 0;
          *(_QWORD *)(v5 + 8) = 0;
          *(_QWORD *)(v5 + 16) = 0;
          *(_QWORD *)(v5 + 24) = 0;
          v5 = *(_QWORD *)(a1 + 80);
        }
        v15 = v5 + 32;
        *(_QWORD *)(a1 + 80) = v15;
      }
      *(_DWORD *)(v15 - 32) = a2;
      v7 = *(_QWORD *)(a1 + 80) - 32LL;
      v10 = *(_QWORD *)(v7 + 16);
      if ( v10 != *(_QWORD *)(v7 + 24) )
        goto LABEL_10;
LABEL_19:
      sub_1512F90((char **)(v7 + 8), (char *)v10, a3);
      v13 = *(_QWORD *)(v7 + 16);
      return (unsigned int)((v13 - *(_QWORD *)(v7 + 8)) >> 4) + 3;
    }
    v9 = (__int64)&v6[8 * (unsigned int)(v8 - 1) + 8];
    while ( 1 )
    {
      v7 = (__int64)v6;
      if ( a2 == *v6 )
        break;
      v6 += 8;
      if ( (_DWORD *)v9 == v6 )
        goto LABEL_14;
    }
  }
  v10 = *(_QWORD *)(v7 + 16);
  if ( v10 == *(_QWORD *)(v7 + 24) )
    goto LABEL_19;
LABEL_10:
  if ( v10 )
  {
    v11 = *a3;
    *(_QWORD *)(v10 + 8) = 0;
    *a3 = 0;
    *(_QWORD *)v10 = v11;
    v12 = a3[1];
    a3[1] = 0;
    *(_QWORD *)(v10 + 8) = v12;
    v10 = *(_QWORD *)(v7 + 16);
  }
  v13 = v10 + 16;
  *(_QWORD *)(v7 + 16) = v13;
  return (unsigned int)((v13 - *(_QWORD *)(v7 + 8)) >> 4) + 3;
}

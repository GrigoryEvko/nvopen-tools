// Function: sub_A84C60
// Address: 0xa84c60
//
__int64 __fastcall sub_A84C60(int a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v6; // ecx
  __int64 v7; // rsi
  int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax

  if ( a1 != 49 )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  v6 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v6 - 17) <= 1 )
  {
    v4 = **(_QWORD **)(v4 + 16);
    LOBYTE(v6) = *(_BYTE *)(v4 + 8);
  }
  if ( (_BYTE)v6 != 14 )
    return 0;
  v7 = a3;
  v8 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
  {
    v9 = *(__int64 **)(a3 + 16);
    v7 = *v9;
    LOBYTE(v8) = *(_BYTE *)(*v9 + 8);
  }
  if ( (_BYTE)v8 != 14 )
    return 0;
  v10 = *(_DWORD *)(v7 + 8) >> 8;
  if ( (_DWORD)v10 == *(_DWORD *)(v4 + 8) >> 8 )
    return 0;
  v11 = sub_BD5C60(a2, v7, v10);
  v12 = sub_BCB2E0(v11);
  v13 = sub_AD4C50(a2, v12, 0);
  return sub_AD4C70(v13, a3, 0);
}

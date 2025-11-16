// Function: sub_A84B50
// Address: 0xa84b50
//
__int64 __fastcall sub_A84B50(int a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE v17[32]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v18; // [rsp-38h] [rbp-38h]

  if ( a1 != 49 )
    return 0;
  *a4 = 0;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
  {
    v7 = **(_QWORD **)(v7 + 16);
    LOBYTE(v8) = *(_BYTE *)(v7 + 8);
  }
  if ( (_BYTE)v8 != 14 )
    return 0;
  v9 = a3;
  v10 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
  {
    v11 = *(__int64 **)(a3 + 16);
    v9 = *v11;
    LOBYTE(v10) = *(_BYTE *)(*v11 + 8);
  }
  if ( (_BYTE)v10 != 14 )
    return 0;
  v12 = *(_DWORD *)(v9 + 8) >> 8;
  if ( (_DWORD)v12 == *(_DWORD *)(v7 + 8) >> 8 )
    return 0;
  v13 = sub_BD5C60(a2, v9, v12);
  v14 = sub_BCB2E0(v13);
  v18 = 257;
  v15 = sub_B51D30(47, a2, v14, v17, 0, 0);
  *a4 = v15;
  v18 = 257;
  return sub_B51D30(48, v15, a3, v17, 0, 0);
}

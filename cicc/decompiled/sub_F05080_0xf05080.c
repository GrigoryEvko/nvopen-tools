// Function: sub_F05080
// Address: 0xf05080
//
__int64 __fastcall sub_F05080(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r9
  int v5; // edx
  __int64 v6; // r9
  int v7; // edx
  __int64 v8; // r9
  int v9; // edx
  int v10; // ecx
  __int64 v11; // r9
  int v12; // ecx
  _BYTE *v13; // [rsp+0h] [rbp-20h] BYREF
  __int64 v14; // [rsp+8h] [rbp-18h]
  int v15; // [rsp+10h] [rbp-10h] BYREF
  int v16; // [rsp+14h] [rbp-Ch] BYREF
  int v17; // [rsp+18h] [rbp-8h] BYREF
  int v18; // [rsp+1Ch] [rbp-4h] BYREF

  v13 = a2;
  v14 = a3;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  result = sub_F04E10((__int64 *)&v13, &v15);
  if ( (_BYTE)result )
    return 1;
  if ( !v14 )
  {
    v5 = v15;
    *(_QWORD *)(v4 + 4) = 0;
    *(_DWORD *)(v4 + 12) = 0;
    *(_DWORD *)v4 = v5;
    return result;
  }
  if ( *v13 != 46 )
    return 1;
  ++v13;
  --v14;
  result = sub_F04E10((__int64 *)&v13, &v16);
  if ( (_BYTE)result )
    return 1;
  if ( !v14 )
  {
    v7 = v15;
    *(_DWORD *)(v6 + 12) = 0;
    *(_DWORD *)v6 = v7;
    *(_QWORD *)(v6 + 4) = v16 | 0x80000000;
    return result;
  }
  result = 1;
  if ( *v13 != 46 )
    return result;
  ++v13;
  --v14;
  result = sub_F04E10((__int64 *)&v13, &v17);
  if ( (_BYTE)result )
    return 1;
  if ( !v14 )
  {
    v9 = v15;
    v10 = v16;
    *(_DWORD *)(v8 + 12) = 0;
    *(_DWORD *)v8 = v9;
    *(_QWORD *)(v8 + 4) = v10 & 0x7FFFFFFF | 0x8000000080000000LL | ((unsigned __int64)(v17 & 0x7FFFFFFF) << 32);
    return result;
  }
  result = 1;
  if ( *v13 == 46 )
  {
    ++v13;
    --v14;
    result = sub_F04E10((__int64 *)&v13, &v18);
    if ( !(_BYTE)result && !v14 )
    {
      v12 = v16;
      *(_DWORD *)v11 = v15;
      *(_QWORD *)(v11 + 4) = v12 & 0x7FFFFFFF | 0x8000000080000000LL | ((unsigned __int64)(v17 & 0x7FFFFFFF) << 32);
      *(_DWORD *)(v11 + 12) = v18 | 0x80000000;
      return result;
    }
    return 1;
  }
  return result;
}

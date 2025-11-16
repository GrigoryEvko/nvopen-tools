// Function: sub_FCD870
// Address: 0xfcd870
//
__int64 __fastcall sub_FCD870(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  unsigned __int8 v4; // al
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // edx
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // [rsp-38h] [rbp-38h] BYREF
  __int64 v15; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)a1 <= 0x15u )
    return 0;
  result = 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v3 + 8) != 7 )
  {
    if ( sub_BCAC40(*(_QWORD *)(a1 + 8), 1) )
      return 0x100000000LL;
    v4 = *(_BYTE *)(v3 + 8);
    if ( v4 == 15 )
    {
      v5 = sub_AE4AC0(a2, v3);
      v6 = *(_QWORD *)v5;
      LOBYTE(v5) = *(_BYTE *)(v5 + 8);
      v14 = v6;
      LOBYTE(v15) = v5;
      v7 = sub_CA1930(&v14);
    }
    else if ( *(_BYTE *)a1 == 84 && v4 == 17 )
    {
      v8 = *(_DWORD *)(v3 + 32);
      v9 = sub_9208B0(a2, *(_QWORD *)(v3 + 24));
      v15 = v10;
      v14 = (unsigned __int64)(v9 + 7) >> 3;
      v11 = sub_CA1930(&v14);
      if ( v11 > 4 )
        v11 = 4;
      v7 = v8 * v11;
    }
    else
    {
      if ( v4 != 12
        && v4 > 3u
        && v4 != 5
        && (v4 & 0xFD) != 4
        && (v4 & 0xFB) != 0xA
        && ((unsigned __int8)(v4 - 15) > 3u && v4 != 20 || !(unsigned __int8)sub_BCEBA0(v3, 0)) )
      {
        return 1;
      }
      v12 = sub_9208B0(a2, v3);
      v15 = v13;
      v14 = (unsigned __int64)(v12 + 7) >> 3;
      v7 = sub_CA1930(&v14);
    }
    result = (unsigned int)(v7 / 4);
    if ( v7 <= 3 )
      return 1;
  }
  return result;
}

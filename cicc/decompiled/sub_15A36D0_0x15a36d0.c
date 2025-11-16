// Function: sub_15A36D0
// Address: 0x15a36d0
//
__int64 __fastcall sub_15A36D0(unsigned __int16 a1, _QWORD *a2, _QWORD *a3, char a4)
{
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-60h] BYREF
  __int128 v14; // [rsp+10h] [rbp-50h]
  __int128 v15; // [rsp+20h] [rbp-40h]
  __int128 v16; // [rsp+30h] [rbp-30h]

  result = sub_1587A80(a1, a2, a3);
  if ( !result && !a4 )
  {
    *(_QWORD *)&v14 = 52;
    v13[0] = a2;
    v13[1] = a3;
    WORD1(v14) = a1;
    *((_QWORD *)&v14 + 1) = v13;
    v15 = 2u;
    v16 = 0u;
    v7 = sub_16498A0(a2);
    v10 = sub_1643320(v7);
    v12 = *a2;
    if ( *(_BYTE *)(*a2 + 8LL) == 16 )
    {
      v10 = sub_16463B0(v10, *(_QWORD *)(v12 + 32));
      v12 = *a2;
    }
    return sub_15A2780(**(_QWORD **)v12 + 1776LL, v10, v8, v9, v10, v11, v14, v15, v16);
  }
  return result;
}

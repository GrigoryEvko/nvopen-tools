// Function: sub_11DDD00
// Address: 0x11ddd00
//
__int64 __fastcall sub_11DDD00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 result; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // [rsp+8h] [rbp-48h]
  __int64 *v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  __int64 v14; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v5 == 17 )
  {
    LODWORD(v13) = 0;
    sub_11DA4B0(a2, (int *)&v13, 1);
    v13 = 0;
    v14 = 0;
    if ( !(unsigned __int8)sub_98B0F0(v4, &v13, 1u) )
    {
      if ( *(_DWORD *)(v5 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v5 + 24) )
          return 0;
      }
      else
      {
        v11 = *(_DWORD *)(v5 + 32);
        if ( v11 != (unsigned int)sub_C444A0(v5 + 24) )
          return 0;
      }
      result = sub_11CA130(v4, 0, a3, *(__int64 **)(a1 + 24));
      if ( result )
      {
        if ( *(_BYTE *)result != 85 )
          return result;
LABEL_13:
        *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
        return result;
      }
      return 0;
    }
  }
  else
  {
    LODWORD(v13) = 0;
    sub_11DA4B0(a2, (int *)&v13, 1);
    v13 = 0;
    v14 = 0;
    if ( !(unsigned __int8)sub_98B0F0(v4, &v13, 1u) )
      return 0;
  }
  v12 = *(__int64 **)(a1 + 24);
  v7 = sub_B43CA0(a2);
  LODWORD(v12) = sub_97FA80(*v12, v7);
  v8 = (_QWORD *)sub_BD5C60(a2);
  v9 = sub_BCCE00(v8, (unsigned int)v12);
  v10 = sub_AD64C0(v9, v14 + 1, 0);
  result = sub_11CA840(v4, v5, v10, a3, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    goto LABEL_13;
  return result;
}

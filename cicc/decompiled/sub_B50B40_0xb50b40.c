// Function: sub_B50B40
// Address: 0xb50b40
//
__int64 __fastcall sub_B50B40(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // edx
  int v4; // r9d
  char v5; // cl
  __int64 v7; // rbx
  char v8; // dl
  __int64 v9; // rax
  char v10; // dl
  char v11; // [rsp+18h] [rbp-28h]

  v3 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v3 == 7 )
    return 0;
  if ( v3 == 13 )
    return 0;
  v4 = *(unsigned __int8 *)(a2 + 8);
  v5 = *(_BYTE *)(a2 + 8);
  LOBYTE(v2) = (_BYTE)v4 != 13 && (_BYTE)v4 != 7;
  if ( !(_BYTE)v2 )
    return 0;
  if ( a1 != a2 )
  {
    if ( (unsigned int)(v3 - 17) <= 1
      && (unsigned int)(v4 - 17) <= 1
      && ((_BYTE)v3 == 18) == (v5 == 18)
      && *(_DWORD *)(a1 + 32) == *(_DWORD *)(a2 + 32) )
    {
      a2 = *(_QWORD *)(a2 + 24);
      a1 = *(_QWORD *)(a1 + 24);
      v5 = *(_BYTE *)(a2 + 8);
    }
    if ( v5 == 14 && *(_BYTE *)(a1 + 8) == 14 )
    {
      LOBYTE(v2) = *(_DWORD *)(a2 + 8) >> 8 == *(_DWORD *)(a1 + 8) >> 8;
      return v2;
    }
    v7 = sub_BCAE30(a1);
    v11 = v8;
    v9 = sub_BCAE30(a2);
    if ( !v9 || !v7 || v7 != v9 || v11 != v10 )
      return 0;
  }
  return v2;
}

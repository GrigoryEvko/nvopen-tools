// Function: sub_255E3B0
// Address: 0x255e3b0
//
__int64 __fastcall sub_255E3B0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r12
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rdx

  v3 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  result = *v3;
  if ( (unsigned __int8)result <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC2D00((__int64)v3) )
    {
LABEL_3:
      result = *(unsigned __int8 *)(a1 + 96);
      *(_BYTE *)(a1 + 97) = result;
      return result;
    }
LABEL_10:
    result = *(unsigned __int8 *)(a1 + 97);
    *(_BYTE *)(a1 + 96) = result;
    return result;
  }
  if ( (unsigned __int8)result <= 0x1Cu )
    return result;
  if ( (unsigned __int8)(result - 34) > 0x33u || (v6 = 0x8000000000041LL, !_bittest64(&v6, (unsigned int)(result - 34))) )
  {
LABEL_19:
    v8 = *(_QWORD *)(a2 + 208);
    v9 = sub_B43CB0((__int64)v3);
    v10 = sub_255E1F0(*(_QWORD *)(v8 + 240), v9, 0);
    if ( !v10 )
      goto LABEL_3;
    result = sub_E387E0(v10, *((_QWORD *)v3 + 5));
    if ( result )
      goto LABEL_3;
    return result;
  }
  if ( (_DWORD)result == 40 )
  {
    v7 = 32LL * (unsigned int)sub_B491D0((__int64)v3);
  }
  else
  {
    v7 = 0;
    if ( (_DWORD)result != 85 )
    {
      v7 = 64;
      if ( (_DWORD)result != 34 )
        BUG();
    }
  }
  if ( (v3[7] & 0x80u) == 0 )
    goto LABEL_6;
  v11 = sub_BD2BC0((__int64)v3);
  v13 = v11 + v12;
  if ( (v3[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v13 >> 4) )
      goto LABEL_29;
LABEL_6:
    v5 = 0;
    goto LABEL_7;
  }
  if ( !(unsigned int)((v13 - sub_BD2BC0((__int64)v3)) >> 4) )
    goto LABEL_6;
  if ( (v3[7] & 0x80u) == 0 )
LABEL_29:
    BUG();
  v14 = *(_DWORD *)(sub_BD2BC0((__int64)v3) + 8);
  if ( (v3[7] & 0x80u) == 0 )
    BUG();
  v15 = sub_BD2BC0((__int64)v3);
  v5 = 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
LABEL_7:
  result = (32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF) - 32 - v7 - v5) >> 5;
  if ( !(_DWORD)result )
  {
    result = sub_B46970(v3);
    if ( !(_BYTE)result )
    {
      result = sub_B46420((__int64)v3);
      if ( !(_BYTE)result )
        goto LABEL_10;
    }
  }
  if ( *v3 > 0x1Cu )
    goto LABEL_19;
  return result;
}

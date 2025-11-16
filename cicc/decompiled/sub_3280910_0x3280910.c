// Function: sub_3280910
// Address: 0x3280910
//
bool __fastcall sub_3280910(_WORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // bl
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  bool result; // al
  _QWORD v11[4]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+20h] [rbp-30h]
  __int64 v13; // [rsp+28h] [rbp-28h]

  v11[0] = a2;
  v11[1] = a3;
  if ( (_WORD)a2 )
  {
    if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
      goto LABEL_10;
    v3 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
    v6 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
    LOWORD(v4) = *a1;
    if ( !*a1 )
      goto LABEL_3;
LABEL_12:
    if ( (_WORD)v4 != 1 && (unsigned __int16)(v4 - 504) > 7u )
    {
      v9 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v4 - 16];
      LOBYTE(v4) = byte_444C4A0[16 * (unsigned __int16)v4 - 8];
      goto LABEL_4;
    }
LABEL_10:
    BUG();
  }
  v12 = sub_3007260((__int64)v11);
  v3 = v12;
  LOWORD(v4) = *a1;
  v13 = v5;
  v6 = v5;
  if ( (_WORD)v4 )
    goto LABEL_12;
LABEL_3:
  v7 = sub_3007260((__int64)a1);
  v4 = v8;
  v11[2] = v7;
  v9 = v7;
  v11[3] = v4;
LABEL_4:
  if ( !(_BYTE)v4 )
    return v9 <= v3;
  result = v6;
  if ( v6 )
    return v9 <= v3;
  return result;
}

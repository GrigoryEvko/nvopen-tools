// Function: sub_3280240
// Address: 0x3280240
//
bool __fastcall sub_3280240(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v3; // bx
  bool result; // al
  __int64 v5; // r14
  __int64 v6; // rdx
  char v7; // r13
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // cl
  _QWORD v13[4]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-30h]
  __int64 v15; // [rsp+28h] [rbp-28h]

  v3 = *(_WORD *)a1;
  v13[0] = a2;
  v13[1] = a3;
  if ( v3 != (_WORD)a2 )
  {
    if ( (_WORD)a2 )
    {
      if ( (_WORD)a2 == 1 || (unsigned __int16)(a2 - 504) <= 7u )
        goto LABEL_11;
      v5 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a2 - 16];
      v7 = byte_444C4A0[16 * (unsigned __int16)a2 - 8];
      if ( !v3 )
        goto LABEL_6;
LABEL_13:
      if ( v3 != 1 && (unsigned __int16)(v3 - 504) > 7u )
      {
        v11 = *(_QWORD *)&byte_444C4A0[16 * v3 - 16];
        v12 = byte_444C4A0[16 * v3 - 8];
LABEL_7:
        result = 0;
        if ( v5 == v11 )
          return v12 == v7;
        return result;
      }
LABEL_11:
      BUG();
    }
LABEL_5:
    v14 = sub_3007260((__int64)v13);
    v5 = v14;
    v15 = v6;
    v7 = v6;
    if ( !v3 )
    {
LABEL_6:
      v8 = sub_3007260(a1);
      v10 = v9;
      v13[2] = v8;
      v11 = v8;
      v13[3] = v10;
      v12 = v10;
      goto LABEL_7;
    }
    goto LABEL_13;
  }
  result = 1;
  if ( !v3 && *(_QWORD *)(a1 + 8) != a3 )
    goto LABEL_5;
  return result;
}

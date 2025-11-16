// Function: sub_12FC4A0
// Address: 0x12fc4a0
//
unsigned __int64 __fastcall sub_12FC4A0(
        char a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5)
{
  unsigned __int64 result; // rax
  __int64 v7; // r12
  int v8; // r14d
  char v9; // r15
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // [rsp+8h] [rbp-58h]
  __int64 v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v13; // [rsp+18h] [rbp-48h]
  unsigned __int64 v14; // [rsp+20h] [rbp-40h]

  result = a4;
  v7 = a5;
  v8 = unk_4F968EB;
  v9 = unk_4F968EB & 0xFB;
  if ( (unk_4F968EB & 0xFB) != 0 )
    LOBYTE(v10) = unk_4F968EB == (a1 == 0) + 2 && a5 != 0;
  else
    v10 = a5 >> 63;
  if ( (unsigned int)a2 <= 0x7FFC )
    goto LABEL_13;
  if ( a2 < 0 )
  {
    if ( !unk_4C6F00D
      || (unsigned __int8)v10 ^ 1 | (a2 < -1)
      || a3 <= 0x1FFFFFFFFFFFELL
      || a3 == 0x1FFFFFFFFFFFFLL && a4 != -1 )
    {
      sub_12FB780((unsigned __int64 *)&v12, a3, a4, a5, -a2);
      v7 = v12;
      result = v13;
      a3 = v14;
      if ( v12 )
      {
        v11 = v13;
        sub_12F9B70(2);
        result = v11;
        if ( !v9 )
        {
          unk_4F968EA |= 1u;
          if ( v7 < 0 )
            return (result + 1)
                 & ~(unsigned __int64)(((v7 & 0x7FFFFFFFFFFFFFFFLL) == 0) & (unsigned __int8)((_BYTE)v8 == 0));
          goto LABEL_19;
        }
LABEL_12:
        LOBYTE(v10) = (unsigned __int8)v8 == (a1 == 0) + 2 && v7 != 0;
        goto LABEL_13;
      }
    }
    else
    {
      sub_12FB780((unsigned __int64 *)&v12, a3, a4, a5, 1u);
      v7 = v12;
      result = v13;
      a3 = v14;
    }
    if ( !v9 )
    {
      v10 = (unsigned __int64)v7 >> 63;
      goto LABEL_13;
    }
    goto LABEL_12;
  }
  if ( a2 != 32765 )
  {
LABEL_31:
    sub_12F9B70(5);
    if ( !v9 || v8 == (a1 == 0) + 2 )
      return 0;
    else
      return -1;
  }
  if ( a3 != 0x1FFFFFFFFFFFFLL || a4 != -1 )
  {
LABEL_13:
    if ( v7 )
    {
      unk_4F968EA |= 1u;
      if ( (_BYTE)v8 == 6 )
        goto LABEL_17;
    }
    if ( (_BYTE)v10 )
      return (result + 1) & ~(unsigned __int64)(((v7 & 0x7FFFFFFFFFFFFFFFLL) == 0) & (unsigned __int8)((_BYTE)v8 == 0));
LABEL_19:
    if ( !(a3 | result) )
      return 0;
    return result;
  }
  if ( (_BYTE)v10 )
    goto LABEL_31;
  result = -1;
  if ( a5 )
  {
    unk_4F968EA |= 1u;
    if ( unk_4F968EB == 6 )
LABEL_17:
      result |= 1u;
  }
  return result;
}

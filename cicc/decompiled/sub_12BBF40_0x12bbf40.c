// Function: sub_12BBF40
// Address: 0x12bbf40
//
__int64 __fastcall sub_12BBF40(__int64 a1, unsigned int a2, char *a3)
{
  char v4; // bl
  __int64 v5; // r15
  unsigned int v6; // r12d

  v4 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v5 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v6 = 5;
      goto LABEL_17;
    }
    *(_QWORD *)(a1 + 88) = 0;
    **(_BYTE **)(a1 + 80) = 0;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
    {
      v6 = 8;
      goto LABEL_17;
    }
    v4 = 1;
LABEL_13:
    if ( (unsigned __int8)sub_12BB580(a1, a2, a3) )
    {
      v6 = sub_225F500(a1, (unsigned int)dword_4C6F008, a2, a3);
      if ( !v4 )
        return v6;
    }
    else
    {
      v6 = sub_12C5890(a1, (unsigned int)dword_4C6F008, a2, a3);
      if ( !v4 )
        return v6;
    }
LABEL_17:
    sub_16C30E0(v5);
    return v6;
  }
  if ( !qword_4F92D80 )
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  v5 = qword_4F92D80;
  if ( !a1 )
    return 5;
  *(_QWORD *)(a1 + 88) = 0;
  **(_BYTE **)(a1 + 80) = 0;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
    goto LABEL_13;
  return 8;
}

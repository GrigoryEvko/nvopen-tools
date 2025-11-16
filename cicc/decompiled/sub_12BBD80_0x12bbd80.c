// Function: sub_12BBD80
// Address: 0x12bbd80
//
__int64 __fastcall sub_12BBD80(__int64 a1, unsigned int a2, char *a3)
{
  char v4; // bl
  __int64 v5; // r15
  _BYTE *v6; // rax
  unsigned int v7; // r12d
  _BYTE *v8; // rax

  v4 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v5 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v7 = 5;
      goto LABEL_17;
    }
    *(_QWORD *)(a1 + 56) = 0;
    **(_BYTE **)(a1 + 48) = 0;
    v8 = *(_BYTE **)(a1 + 80);
    *(_QWORD *)(a1 + 88) = 0;
    *v8 = 0;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
    {
      v7 = 8;
      goto LABEL_17;
    }
    v4 = 1;
LABEL_13:
    if ( (unsigned __int8)sub_12BB580(a1, a2, a3) )
    {
      v7 = sub_225F3E0(a1, (unsigned int)dword_4C6F008, a2, a3);
      if ( !v4 )
        return v7;
    }
    else
    {
      v7 = sub_12C5770(a1, (unsigned int)dword_4C6F008, a2, a3);
      if ( !v4 )
        return v7;
    }
LABEL_17:
    sub_16C30E0(v5);
    return v7;
  }
  if ( !qword_4F92D80 )
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  v5 = qword_4F92D80;
  if ( !a1 )
    return 5;
  *(_QWORD *)(a1 + 56) = 0;
  **(_BYTE **)(a1 + 48) = 0;
  v6 = *(_BYTE **)(a1 + 80);
  *(_QWORD *)(a1 + 88) = 0;
  *v6 = 0;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
    goto LABEL_13;
  return 8;
}

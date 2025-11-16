// Function: sub_869D70
// Address: 0x869d70
//
unsigned int *__fastcall sub_869D70(__int64 a1, char a2)
{
  unsigned int *result; // rax
  unsigned int v3; // r12d
  _BYTE *v4; // r14
  _BYTE *v5; // rax
  _BYTE *v6; // rax
  _BYTE *v7; // rax
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = &dword_4F04C3C;
  v3 = dword_4F04C3C;
  if ( dword_4F04C3C )
    return result;
  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 && dword_4F07270[0] != dword_4F073B8[0] )
  {
    sub_7296C0(v8);
    v4 = sub_727170();
    *(_QWORD *)v4 = *(_QWORD *)&dword_4F063F8;
    if ( a2 != 21 )
    {
      v4[8] = a2;
LABEL_9:
      *((_QWORD *)v4 + 2) = a1;
      v6 = sub_727090();
      v6[16] = 54;
      *((_QWORD *)v6 + 3) = v4;
      sub_869970((__int64)v6);
      return sub_729730(v8[0]);
    }
    if ( *(_BYTE *)(a1 + 40) != 11 || *(_DWORD *)a1 )
    {
      v4[8] = 21;
      goto LABEL_9;
    }
    v3 = 1;
    goto LABEL_15;
  }
  v4 = sub_727170();
  *(_QWORD *)v4 = *(_QWORD *)&dword_4F063F8;
  if ( a2 != 21 )
  {
    v4[8] = a2;
LABEL_5:
    *((_QWORD *)v4 + 2) = a1;
    v5 = sub_727090();
    v5[16] = 54;
    *((_QWORD *)v5 + 3) = v4;
    return (unsigned int *)sub_869970((__int64)v5);
  }
  if ( *(_BYTE *)(a1 + 40) != 11 || *(_DWORD *)a1 )
  {
    v4[8] = 21;
    goto LABEL_5;
  }
LABEL_15:
  v4[8] = a2;
  *((_QWORD *)v4 + 2) = a1;
  *(_QWORD *)v4 = *(_QWORD *)&dword_4F077C8;
  v7 = sub_727090();
  v7[16] = 54;
  *((_QWORD *)v7 + 3) = v4;
  result = (unsigned int *)sub_869970((__int64)v7);
  if ( v3 )
    return sub_729730(v8[0]);
  return result;
}

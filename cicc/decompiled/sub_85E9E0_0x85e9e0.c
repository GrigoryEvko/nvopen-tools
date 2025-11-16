// Function: sub_85E9E0
// Address: 0x85e9e0
//
__int16 __fastcall sub_85E9E0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx

  if ( a1 )
  {
    v2 = *(_QWORD *)(a1 + 120);
    if ( (unsigned int)sub_8D3410(v2) )
    {
      v2 = sub_8D40F0(v2);
      if ( (*(_BYTE *)(v2 + 140) & 0xFB) != 8 )
        goto LABEL_4;
    }
    else if ( (*(_BYTE *)(v2 + 140) & 0xFB) != 8 )
    {
      goto LABEL_4;
    }
    LOWORD(v3) = sub_8D4C10(v2, dword_4F077C4 != 2);
    if ( (v3 & 1) != 0 )
      return v3;
  }
LABEL_4:
  LOWORD(v3) = qword_4F04C50;
  v4 = *(_QWORD *)(qword_4F04C50 + 32LL);
  if ( *(char *)(v4 + 192) < 0 )
  {
    LOWORD(v3) = *(_WORD *)(v4 + 202) & 0x2002;
    if ( (_WORD)v3 == 0x2000 && !*(_BYTE *)(v4 + 172) )
    {
      v3 = qword_4F5FD10;
      if ( qword_4F5FD10 )
        qword_4F5FD10 = *(_QWORD *)qword_4F5FD10;
      else
        v3 = sub_823970(32);
      v5 = qword_4F5FD18;
      *(_QWORD *)(v3 + 8) = v4;
      *(_QWORD *)v3 = v5;
      qword_4F5FD18 = v3;
      *(_QWORD *)(v3 + 16) = *a2;
      *(_DWORD *)(v3 + 24) = a1 != 0;
    }
  }
  return v3;
}

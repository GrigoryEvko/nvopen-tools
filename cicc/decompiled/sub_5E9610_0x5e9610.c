// Function: sub_5E9610
// Address: 0x5e9610
//
__int64 __fastcall sub_5E9610(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rcx
  __int64 result; // rax
  __int64 v6; // rax

  v3 = qword_4CF8008;
  if ( (unsigned int)sub_5E4740(a1, a2, a3, *(_QWORD *)(qword_4CF8008 + 16))
    || !v4
    || (*(_BYTE *)(v4 + 81) & 0x20) != 0
    || !*(_QWORD *)(v3 + 160) && !*(_QWORD *)(v3 + 128) && (*(_BYTE *)(v3 + 184) & 8) == 0 )
  {
    if ( dword_4F04C64 != -1
      && (v6 = unk_4F04C68 + 776LL * dword_4F04C64, (*(_BYTE *)(v6 + 7) & 1) != 0)
      && (unk_4F04C44 != -1 || (*(_BYTE *)(v6 + 6) & 2) != 0)
      || (*(_BYTE *)(v3 + 89) & 8) != 0 )
    {
      if ( (*(_BYTE *)(v3 + 184) & 2) != 0 )
      {
LABEL_12:
        result = qword_4CF8000;
        *(_QWORD *)(v3 + 128) = 0;
        *(_QWORD *)v3 = result;
        qword_4CF8000 = v3;
        goto LABEL_13;
      }
    }
    else
    {
      sub_87E280(v3 + 32);
      v3 = qword_4CF8008;
      if ( (*(_BYTE *)(qword_4CF8008 + 184) & 2) != 0 )
        goto LABEL_12;
    }
    sub_679050(*(_QWORD *)(v3 + 128));
    goto LABEL_12;
  }
  result = sub_5E9580(v3);
LABEL_13:
  qword_4CF8008 = 0;
  return result;
}

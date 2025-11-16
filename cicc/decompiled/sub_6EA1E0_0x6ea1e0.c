// Function: sub_6EA1E0
// Address: 0x6ea1e0
//
__int64 __fastcall sub_6EA1E0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // r12d

  v1 = *(_QWORD *)(a1 + 120);
  if ( ((unsigned int)sub_8D2930(v1)
     && (*(_BYTE *)(v1 + 140) & 0xFB) == 8
     && (sub_8D4C10(v1, dword_4F077C4 != 2) & 1) != 0
     || (v2 = sub_8D3D40(v1)) != 0
     || (*(_BYTE *)(a1 + 172) & 8) != 0 && (unsigned int)sub_8D4160(v1)
     || word_4D04898 && (unsigned int)sub_8D32E0(v1)
     || (dword_4F077BC || dword_4F077C0 && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x1387Fu)
     && (unsigned int)sub_8D3350(v1)
     && (*(_BYTE *)(v1 + 140) & 0xFB) == 8
     && (sub_8D4C10(v1, dword_4F077C4 != 2) & 1) != 0)
    && (v2 = 1, (*(_BYTE *)(v1 + 140) & 0xFB) == 8) )
  {
    return ((unsigned __int8)((unsigned int)sub_8D4C10(v1, dword_4F077C4 != 2) >> 1) ^ 1) & 1;
  }
  else
  {
    return v2;
  }
}

// Function: sub_8E43E0
// Address: 0x8e43e0
//
__int64 __fastcall sub_8E43E0(_BYTE *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  unsigned int v4; // r13d
  __int64 v5; // rax
  int v6; // eax
  __int64 v8; // rax

  if ( (unsigned __int8)(a1[140] - 9) > 2u )
    sub_721090();
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  v3 = *(_QWORD *)(v2 + 16);
  v4 = sub_8E3AD0((__int64)a1, a2);
  if ( (unsigned int)sub_8E3AD0((__int64)a1, a2) )
  {
    if ( v3 )
    {
      v4 = 1;
      goto LABEL_5;
    }
    if ( (*(_BYTE *)(v2 + 176) & 1) == 0 && (*(_QWORD *)(v2 + 16) || !*(_QWORD *)(v2 + 8)) )
      return 1;
    v8 = sub_5EB340(v2);
    v3 = v8;
    if ( !v8 || (*(_BYTE *)(*(_QWORD *)(v8 + 88) + 194LL) & 2) == 0 )
      return 0;
  }
  if ( !v4 )
    return v4;
  if ( v3 )
  {
LABEL_5:
    v5 = *(_QWORD *)(v3 + 88);
    if ( (*(_BYTE *)(v5 + 206) & 0x10) != 0 )
    {
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 && !qword_4F077A8 )
          return 0;
      }
      else if ( !(_DWORD)qword_4F077B4 )
      {
        return 0;
      }
    }
    if ( (*(_BYTE *)(v3 + 104) & 1) != 0 )
    {
      v6 = sub_8796F0(v3);
    }
    else
    {
      if ( *(_BYTE *)(v3 + 80) == 20 )
        v5 = *(_QWORD *)(v5 + 176);
      v6 = (*(_BYTE *)(v5 + 208) & 4) != 0;
    }
    if ( !v6 )
      return v4;
    return 0;
  }
  return v4;
}

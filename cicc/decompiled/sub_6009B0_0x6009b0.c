// Function: sub_6009B0
// Address: 0x6009b0
//
__int64 __fastcall sub_6009B0(__int64 a1, __int64 a2, int a3)
{
  unsigned int v4; // r12d
  unsigned int v6; // r14d
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v13; // r12

  v4 = 1;
  if ( (*(_BYTE *)(a1 + 195) & 3) != 1 )
    v4 = (*(_BYTE *)(a1 + 206) & 8) != 0;
  if ( (*(_BYTE *)(a2 + 176) & 0x10) == 0 )
  {
LABEL_7:
    v7 = 0;
    v8 = a2;
    sub_8646E0(a2, 0);
    if ( (*(_BYTE *)(a2 + 177) & 0x20) == 0 )
    {
      v7 = v4;
      v8 = a2;
      if ( (unsigned int)sub_6007F0(a2, v4) )
      {
        if ( !a3 || (v13 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 8LL)) == 0 )
        {
LABEL_14:
          v6 = 1;
          goto LABEL_9;
        }
        while ( 1 )
        {
          v8 = *(_QWORD *)(v13 + 40);
          v7 = a2;
          if ( !(unsigned int)sub_600680(v8, a2) )
            break;
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            goto LABEL_14;
        }
      }
    }
    v6 = 0;
LABEL_9:
    sub_866010(v8, v7, v9, v10, v11);
    return v6;
  }
  v6 = dword_4F077BC;
  if ( !dword_4F077BC )
    return v6;
  v6 = dword_4F077B4;
  if ( !dword_4F077B4 )
  {
    if ( qword_4F077A8 )
      goto LABEL_7;
    return v6;
  }
  return 0;
}

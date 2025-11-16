// Function: sub_989E10
// Address: 0x989e10
//
__int64 __fastcall sub_989E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v7; // r14
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r15d
  int v13; // ebx
  int v14; // eax
  unsigned int v15; // eax
  __int64 v17; // rax

  v7 = a4;
  if ( *(_QWORD *)a5 == sub_C33340(a1, a2, a3, a4, a5) )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a5 + 8) + 20LL) & 8) != 0 )
      goto LABEL_15;
    v9 = sub_C40430(a5);
  }
  else
  {
    if ( (*(_BYTE *)(a5 + 20) & 8) != 0 )
      goto LABEL_15;
    v9 = sub_C33CE0(a5);
  }
  if ( !v9 )
    goto LABEL_15;
  if ( a6 )
  {
    if ( *(_BYTE *)v7 == 85 )
    {
      v17 = *(_QWORD *)(v7 - 32);
      if ( v17 )
      {
        if ( !*(_BYTE *)v17 )
        {
          v10 = *(_QWORD *)(v7 + 80);
          if ( *(_QWORD *)(v17 + 24) == v10
            && *(_DWORD *)(v17 + 36) == 170
            && *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)) )
          {
            if ( (_DWORD)a2 == 12 )
              goto LABEL_33;
            if ( (unsigned int)a2 > 0xC )
              goto LABEL_15;
            if ( (_DWORD)a2 == 3 )
            {
LABEL_33:
              v7 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
              v12 = 243;
              v13 = 780;
              goto LABEL_11;
            }
            if ( (unsigned int)a2 > 2 && ((_DWORD)a2 == 4 || (_DWORD)a2 == 11) )
            {
              v7 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
              v12 = 783;
              v13 = 240;
              goto LABEL_11;
            }
LABEL_15:
            v15 = sub_C414D0(a5);
            sub_989640(a1, a2, a3, v7, v15, a6);
            return a1;
          }
        }
      }
    }
    if ( (_DWORD)a2 == 11 )
      goto LABEL_10;
    if ( (unsigned int)a2 > 0xB )
    {
      if ( (_DWORD)a2 != 12 )
        goto LABEL_15;
      goto LABEL_24;
    }
    goto LABEL_8;
  }
  if ( (_DWORD)a2 != 11 )
  {
    if ( (unsigned int)a2 > 0xB )
    {
      if ( (_DWORD)a2 != 12 )
        goto LABEL_15;
      goto LABEL_24;
    }
LABEL_8:
    if ( (_DWORD)a2 != 3 )
    {
      if ( (_DWORD)a2 == 4 )
        goto LABEL_10;
      goto LABEL_15;
    }
LABEL_24:
    v12 = 255;
    v13 = 768;
    goto LABEL_11;
  }
LABEL_10:
  v12 = 771;
  v13 = 252;
LABEL_11:
  if ( (unsigned __int8)sub_B535C0((unsigned int)a2, a2, v10, v11) )
  {
    v14 = v13;
    v13 = v12;
    v12 = v14;
  }
  *(_DWORD *)a1 = v12;
  *(_DWORD *)(a1 + 4) = v13;
  *(_QWORD *)(a1 + 8) = v7;
  return a1;
}

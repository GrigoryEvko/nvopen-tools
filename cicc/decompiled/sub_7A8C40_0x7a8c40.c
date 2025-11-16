// Function: sub_7A8C40
// Address: 0x7a8c40
//
unsigned __int64 __fastcall sub_7A8C40(__int64 a1, unsigned __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int64 v6; // r15
  __int64 v7; // r8
  __int64 v8; // r14
  _QWORD *v9; // r12
  __int64 j; // rdi
  int v11; // eax
  unsigned __int64 v12; // rdx
  __int64 i; // rdi
  int v15; // [rsp+18h] [rbp-38h]

  if ( !(unsigned int)sub_7A6EA0((unsigned __int64 *)(a1 + 8), (unsigned __int64 *)(a1 + 16), a3)
    && !*(_BYTE *)(a1 + 28) )
  {
    sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
    *(_BYTE *)(a1 + 28) = 1;
    v6 = *(_QWORD *)(a1 + 8);
    if ( a4 )
      goto LABEL_4;
LABEL_43:
    if ( a3 <= *(_DWORD *)(a1 + 24) )
      goto LABEL_34;
    goto LABEL_44;
  }
  v6 = *(_QWORD *)(a1 + 8);
  if ( !a4 )
    goto LABEL_43;
LABEL_4:
  v15 = 0;
  while ( 1 )
  {
    if ( (unsigned int)sub_7A8650((__int64 *)a1, a4, v6) )
      goto LABEL_18;
    v6 = *(_QWORD *)(a1 + 8);
    if ( !dword_4D0425C )
      break;
    if ( *(_WORD *)(a4 + 98) == 1 )
      break;
    v8 = **(_QWORD **)(*(_QWORD *)a1 + 168LL);
    if ( !v8 )
      break;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v8 + 96) & 0x43) != 0x41 || *(_QWORD *)(v8 + 104) != v6 )
        goto LABEL_28;
      v9 = **(_QWORD ***)(*(_QWORD *)(v8 + 40) + 168LL);
      if ( v9 )
        break;
      for ( i = *(_QWORD *)(a4 + 40); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (unsigned int)sub_7A6F40(i, v8, 0, (*(_BYTE *)(a4 + 96) & 2) != 0, v7) )
        goto LABEL_18;
LABEL_28:
      v8 = *(_QWORD *)v8;
      if ( !v8 )
        goto LABEL_29;
    }
    while ( 1 )
    {
      if ( !**(_QWORD **)(v9[5] + 168LL) )
      {
        for ( j = *(_QWORD *)(a4 + 40); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (unsigned int)sub_7A6F40(j, (__int64)v9, 0, (*(_BYTE *)(a4 + 96) & 2) != 0, v7) )
          break;
      }
      v9 = (_QWORD *)*v9;
      if ( !v9 )
        goto LABEL_28;
    }
LABEL_18:
    if ( dword_4D0425C )
    {
      v11 = 1;
      if ( (*(_BYTE *)(a4 + 96) & 2) != 0 )
        v11 = v15;
      v15 = v11;
    }
    if ( (unsigned __int64)a3 > unk_4F06AC0 || (v12 = *(_QWORD *)(a1 + 8), v12 > unk_4F06AC0 - (unsigned __int64)a3) )
    {
      if ( !*(_BYTE *)(a1 + 28) )
      {
        sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
        *(_BYTE *)(a1 + 28) = 1;
LABEL_29:
        v6 = *(_QWORD *)(a1 + 8);
        break;
      }
      v6 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      v6 = v12 + a3;
      *(_QWORD *)(a1 + 8) = v6;
    }
  }
  if ( (*(_BYTE *)(a4 + 96) & 3) != 0 )
    *(_QWORD *)(a1 + 72) = a4;
  if ( *(_DWORD *)(a1 + 24) < a3 && (v15 & 1) == 0 )
LABEL_44:
    *(_DWORD *)(a1 + 24) = a3;
LABEL_34:
  if ( a2 <= unk_4F06AC0 && unk_4F06AC0 - a2 >= v6 )
  {
    *(_QWORD *)(a1 + 8) = v6 + a2;
  }
  else if ( !*(_BYTE *)(a1 + 28) )
  {
    sub_6851C0((dword_4F077C4 != 2) + 103, dword_4F07508);
    *(_BYTE *)(a1 + 28) = 1;
  }
  return v6;
}

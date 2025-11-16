// Function: sub_732EF0
// Address: 0x732ef0
//
_BYTE *__fastcall sub_732EF0(__int64 a1)
{
  _BYTE *v1; // r12
  char v3; // al
  int v5; // r13d
  bool v6; // zf
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  int v10; // r15d
  char v11; // di
  int v12; // esi
  _BYTE *v13; // rax
  _QWORD *v14; // r13
  char v15; // al
  _BYTE *v16; // rax
  __int64 v17; // rdi

  v1 = *(_BYTE **)(a1 + 184);
  if ( v1 )
    return v1;
  v3 = *(_BYTE *)(a1 + 4);
  if ( v3 != 2 && v3 != 15 )
  {
    if ( v3 != 1 )
      return v1;
    v5 = dword_4F07270[0];
    sub_7296B0(unk_4F073B8);
    v1 = sub_726EB0(1, *(_DWORD *)a1, 0);
    sub_7296B0(v5);
    v6 = *(_BYTE *)(a1 - 772) == 1;
    *(_QWORD *)(a1 + 184) = v1;
    if ( v6 )
    {
      v17 = a1 - 776;
      if ( *(_QWORD *)(a1 - 472) )
      {
        *((_QWORD *)v1 + 1) = *(_QWORD *)(*(_QWORD *)(a1 - 464) + 8LL);
        **(_QWORD **)(a1 - 464) = v1;
      }
      else
      {
        *(_QWORD *)(a1 - 472) = v1;
      }
      *(_QWORD *)(a1 - 464) = v1;
      *(_QWORD *)v1 = 0;
      *((_QWORD *)v1 + 2) = sub_732EF0(v17);
    }
    else
    {
      *((_QWORD *)v1 + 2) = 0;
    }
    v7 = *(_QWORD *)(a1 + 208);
    if ( v7 )
    {
      while ( *(_BYTE *)(v7 + 140) == 12 )
        v7 = *(_QWORD *)(v7 + 160);
      *(_QWORD *)(*(_QWORD *)(v7 + 168) + 48LL) = v1;
      *((_QWORD *)v1 + 4) = v7;
    }
    goto LABEL_22;
  }
  v8 = *(int *)(a1 + 552);
  v9 = 0;
  if ( (_DWORD)v8 != -1 )
    v9 = qword_4F04C68[0] + 776 * v8;
  sub_732EF0(v9);
  v10 = dword_4F07270[0];
  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    sub_7296B0(*(_DWORD *)(a1 + 192));
    v16 = sub_726EB0(*(_BYTE *)(a1 + 4), *(_DWORD *)a1, 0);
    *(_QWORD *)(a1 + 184) = v16;
    v14 = v16;
    sub_7296B0(v10);
    v13 = *(_BYTE **)(a1 + 184);
  }
  else
  {
    v11 = *(_BYTE *)(a1 + 4);
    v12 = *(_DWORD *)a1;
    *(_DWORD *)(a1 + 192) = dword_4F07270[0];
    v13 = sub_726EB0(v11, v12, 0);
    *(_QWORD *)(a1 + 184) = v13;
    v14 = v13;
  }
  *((_QWORD *)v13 + 2) = *(_QWORD *)(v9 + 184);
  v15 = *(_BYTE *)(v9 + 4);
  if ( ((v15 - 15) & 0xFD) != 0 && v15 != 2 )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_20;
    goto LABEL_27;
  }
  if ( *(_QWORD *)(a1 - 472) )
  {
    v14[1] = *(_QWORD *)(*(_QWORD *)(a1 - 464) + 8LL);
    **(_QWORD **)(a1 - 464) = v14;
  }
  else
  {
    *(_QWORD *)(a1 - 472) = v14;
  }
  *(_QWORD *)(a1 - 464) = v14;
  *v14 = 0;
  if ( dword_4F077C4 == 2 )
  {
LABEL_27:
    sub_732E60(*(unsigned __int8 **)(a1 + 488), 0x17u, v14);
LABEL_20:
    if ( !v14 )
      return v1;
  }
  v1 = v14;
LABEL_22:
  if ( *((_DWORD *)v1 + 60) == -1 )
    *((_DWORD *)v1 + 60) = 1594008481 * ((a1 - qword_4F04C68[0]) >> 3);
  return v1;
}

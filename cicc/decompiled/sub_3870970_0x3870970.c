// Function: sub_3870970
// Address: 0x3870970
//
char __fastcall sub_3870970(__int64 *a1, _QWORD *a2)
{
  _QWORD *v4; // rax
  char v5; // dl
  __int64 v6; // r13
  __int16 v7; // ax
  unsigned int v8; // r12d
  _QWORD *v9; // rsi
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  __int64 v12; // rsi
  _QWORD *v14; // [rsp+8h] [rbp-28h] BYREF

  v14 = a2;
  v4 = (_QWORD *)a1[12];
  if ( (_QWORD *)a1[13] != v4 )
    goto LABEL_2;
  v9 = &v4[*((unsigned int *)a1 + 29)];
  v10 = *((_DWORD *)a1 + 29);
  if ( v4 == v9 )
  {
LABEL_22:
    if ( v10 < *((_DWORD *)a1 + 28) )
    {
      *((_DWORD *)a1 + 29) = v10 + 1;
      *v9 = a2;
      ++a1[11];
      goto LABEL_6;
    }
LABEL_2:
    LOBYTE(v4) = (unsigned __int8)sub_16CCBA0((__int64)(a1 + 11), (__int64)a2);
    if ( !v5 )
      return (char)v4;
LABEL_6:
    v6 = *a1;
    v7 = *((_WORD *)v14 + 12);
    if ( v7 == 6 )
    {
      v4 = (_QWORD *)v14[5];
      if ( !*((_WORD *)v4 + 12) )
      {
        v4 = (_QWORD *)v4[4];
        v8 = *((_DWORD *)v4 + 8);
        if ( v8 <= 0x40 )
        {
          if ( v4[3] )
            goto LABEL_10;
        }
        else
        {
          LODWORD(v4) = sub_16A57B0((__int64)(v4 + 3));
          if ( v8 != (_DWORD)v4 )
          {
LABEL_10:
            LOBYTE(v4) = sub_1458920((__int64)(a1 + 1), &v14);
            return (char)v4;
          }
        }
      }
    }
    else
    {
      if ( v7 != 7 )
        goto LABEL_10;
      v12 = sub_13A5BC0(v14, *(_QWORD *)v6);
      if ( v14[5] == 2 )
        goto LABEL_10;
      LOBYTE(v4) = sub_146D920(*(_QWORD *)v6, v12, **(_QWORD **)(v14[6] + 32LL));
      if ( (_BYTE)v4 )
        goto LABEL_10;
    }
    *(_BYTE *)(v6 + 8) = 1;
    return (char)v4;
  }
  v11 = 0;
  while ( a2 != (_QWORD *)*v4 )
  {
    if ( *v4 == -2 )
      v11 = v4;
    if ( v9 == ++v4 )
    {
      if ( !v11 )
        goto LABEL_22;
      *v11 = a2;
      --*((_DWORD *)a1 + 30);
      ++a1[11];
      goto LABEL_6;
    }
  }
  return (char)v4;
}

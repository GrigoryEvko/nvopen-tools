// Function: sub_14C48A0
// Address: 0x14c48a0
//
__int64 __fastcall sub_14C48A0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int8 v4; // al
  _BYTE *v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  int v12; // eax

  while ( 1 )
  {
    v3 = *(_QWORD *)a1;
    if ( (unsigned int)a2 >= *(_DWORD *)(*(_QWORD *)a1 + 32LL) )
      return sub_1599EF0(*(_QWORD *)(v3 + 24));
    v4 = a1[16];
    if ( v4 <= 0x10u )
      return sub_15A0A60(a1, a2);
    if ( v4 == 84 )
      break;
    if ( v4 == 85 )
    {
      v11 = *(_QWORD *)(**((_QWORD **)a1 - 9) + 32LL);
      v12 = sub_15FA9D0(*((_QWORD *)a1 - 3), a2);
      a2 = (unsigned int)v12;
      if ( v12 < 0 )
        return sub_1599EF0(*(_QWORD *)(v3 + 24));
      if ( v12 < (int)v11 )
      {
LABEL_17:
        a1 = (_BYTE *)*((_QWORD *)a1 - 9);
      }
      else
      {
        a1 = (_BYTE *)*((_QWORD *)a1 - 6);
        a2 = (unsigned int)(v12 - v11);
      }
    }
    else
    {
      if ( v4 != 35 )
        return 0;
      v6 = (_BYTE *)*((_QWORD *)a1 - 6);
      if ( !v6 )
        return 0;
      v7 = *((_QWORD *)a1 - 3);
      if ( *(_BYTE *)(v7 + 16) > 0x10u )
        return 0;
      v8 = sub_15A0A60(v7, a2);
      if ( !v8 )
        return 0;
      a2 = (unsigned int)a2;
      if ( !(unsigned __int8)sub_1593BB0(v8) )
        return 0;
      a1 = v6;
    }
  }
  v9 = *((_QWORD *)a1 - 3);
  if ( *(_BYTE *)(v9 + 16) != 13 )
    return 0;
  if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    v10 = *(_QWORD *)(v9 + 24);
  else
    v10 = **(_QWORD **)(v9 + 24);
  if ( (_DWORD)a2 != (_DWORD)v10 )
    goto LABEL_17;
  return *((_QWORD *)a1 - 6);
}

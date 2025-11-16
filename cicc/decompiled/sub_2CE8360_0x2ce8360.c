// Function: sub_2CE8360
// Address: 0x2ce8360
//
char __fastcall sub_2CE8360(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  int v4; // eax
  __int64 v5; // rcx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax

  v4 = *a3;
  if ( (unsigned __int8)v4 <= 0x1Cu )
  {
    if ( (_BYTE)v4 != 22 )
    {
      if ( (_BYTE)v4 != 3 )
        return (_BYTE)v4 == 5;
      return 1;
    }
    if ( unk_50142AD && (unsigned __int8)sub_CE9220(a2) && !(unsigned __int8)sub_B2D680((__int64)a3)
      || (unsigned __int8)sub_B2D680((__int64)a3) && !(unsigned __int8)sub_CE9220(a2) )
    {
      return 1;
    }
    v7 = *(_QWORD *)(a1 + 8);
    if ( v7 )
    {
      v8 = (_QWORD *)(v7 + 8);
      v9 = *(_QWORD **)(v7 + 16);
      if ( v9 )
      {
        v10 = v8;
        do
        {
          while ( 1 )
          {
            v11 = v9[2];
            v12 = v9[3];
            if ( v9[4] >= (unsigned __int64)a3 )
              break;
            v9 = (_QWORD *)v9[3];
            if ( !v12 )
              goto LABEL_18;
          }
          v10 = v9;
          v9 = (_QWORD *)v9[2];
        }
        while ( v11 );
LABEL_18:
        if ( v8 != v10 && v10[4] <= (unsigned __int64)a3 )
          return 1;
      }
    }
    return 0;
  }
  if ( (_BYTE)v4 == 77 )
  {
    if ( (_BYTE)qword_5013D28 && unk_50142AD && (unsigned __int8)sub_CE9220(a2) )
      return sub_2CE8060(*((_QWORD *)a3 - 4));
    return 0;
  }
  if ( (unsigned __int8)(v4 - 61) <= 0x20u )
  {
    v5 = 0x102820005LL;
    if ( !_bittest64(&v5, (unsigned int)(v4 - 61)) )
      goto LABEL_5;
    return 1;
  }
  if ( *(_BYTE *)(a1 + 1) && (_BYTE)v4 == 60 )
    return 1;
LABEL_5:
  if ( (_BYTE)v4 == 79 )
    return 1;
  if ( (_BYTE)v4 != 85 )
    return (_BYTE)v4 == 5;
  v13 = *((_QWORD *)a3 - 4);
  if ( v13 && !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *((_QWORD *)a3 + 10) && (*(_BYTE *)(v13 + 33) & 0x20) != 0 )
    return *(_DWORD *)(v13 + 36) == 8170;
  else
    return *(_QWORD *)(a1 + 16) != 0;
}

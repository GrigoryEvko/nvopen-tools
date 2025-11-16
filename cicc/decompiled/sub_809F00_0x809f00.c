// Function: sub_809F00
// Address: 0x809f00
//
void __fastcall sub_809F00(__int64 a1, char a2, _DWORD *a3)
{
  _BOOL4 v3; // eax
  __int64 *i; // r14
  _QWORD *v5; // r13
  _QWORD *v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  size_t v9; // rdx

  if ( *(char *)(a1 + 89) < 0 )
    goto LABEL_23;
  if ( a2 != 6 )
  {
    if ( a2 == 28 )
      v3 = (*(_BYTE *)(a1 + 124) & 0x20) != 0;
    else
      v3 = (*(_BYTE *)(a1 + 201) & 4) != 0;
    if ( !v3 )
      return;
    goto LABEL_6;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u
    && (*(_BYTE *)(a1 + 177) & 0x10) != 0
    && *(char *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 160LL) + 89LL) < 0 )
  {
LABEL_23:
    *a3 = 1;
    return;
  }
  if ( (*(_BYTE *)(a1 + 143) & 0x40) == 0 )
    return;
LABEL_6:
  for ( i = *(__int64 **)(a1 + 104); i; i = (__int64 *)*i )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)i + 8) == 80 )
      {
        v5 = (_QWORD *)i[4];
        if ( v5 )
          break;
      }
      i = (__int64 *)*i;
      if ( !i )
        return;
    }
    do
    {
      while ( 1 )
      {
        v6 = (_QWORD *)qword_4F18BA0;
        v7 = v5[5];
        if ( qword_4F18BA0 )
          break;
LABEL_26:
        sub_5D0DD0(qword_4F18B90, byte_4F18B8C, v7);
        v5 = (_QWORD *)*v5;
        if ( !v5 )
          goto LABEL_18;
      }
      while ( 1 )
      {
        v8 = v6[1];
        if ( v7 == v8 )
          break;
        v9 = *(_QWORD *)(v7 + 176);
        if ( v9 == *(_QWORD *)(v8 + 176) && !memcmp(*(const void **)(v7 + 184), *(const void **)(v8 + 184), v9) )
          break;
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          goto LABEL_26;
      }
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
LABEL_18:
    ;
  }
}

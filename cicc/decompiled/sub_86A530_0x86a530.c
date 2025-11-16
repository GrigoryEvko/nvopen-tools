// Function: sub_86A530
// Address: 0x86a530
//
__int64 __fastcall sub_86A530(_QWORD *a1)
{
  __int64 v2; // rbx
  unsigned __int8 v3; // al
  bool v4; // cc
  _BYTE *v5; // rax
  char v6; // dl
  _QWORD *v8; // r13
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *v11; // r13

  v2 = 6338;
LABEL_2:
  while ( 1 )
  {
    v3 = *((_BYTE *)a1 + 16);
    v4 = v3 <= 0x36u;
    if ( v3 == 54 )
      return sub_86A080(a1);
    while ( 1 )
    {
      if ( !v4 )
      {
        if ( v3 == 58 )
        {
          a1 = (_QWORD *)*a1;
          goto LABEL_2;
        }
        goto LABEL_17;
      }
      if ( v3 != 6 )
        break;
      v5 = (_BYTE *)a1[3];
      v6 = v5[140];
      if ( v6 == 12 )
      {
        v10 = (unsigned __int8)v5[184];
        if ( (unsigned __int8)v10 <= 0xCu && _bittest64(&v2, v10) )
        {
          sub_86A530(*a1);
          v11 = (_QWORD *)*a1;
          sub_86A080(a1);
          a1 = v11;
          goto LABEL_2;
        }
LABEL_17:
        sub_721090();
      }
      if ( (unsigned __int8)(v6 - 9) > 2u && (v6 != 2 || (v5[161] & 8) == 0) )
        goto LABEL_17;
      if ( (char)*(v5 - 8) < 0 )
      {
        v9 = sub_86A500(a1);
        a1 = (_QWORD *)sub_86A730(v9);
        goto LABEL_2;
      }
      a1 = (_QWORD *)sub_869630((__int64)a1, 0);
      v3 = *((_BYTE *)a1 + 16);
      v4 = v3 <= 0x36u;
      if ( v3 == 54 )
        return sub_86A080(a1);
    }
    if ( v3 != 53 )
      goto LABEL_17;
    if ( *(char *)(*(_QWORD *)(a1[3] + 24LL) - 8LL) < 0 )
    {
      a1 = (_QWORD *)sub_86A730(a1);
    }
    else
    {
      v8 = (_QWORD *)*a1;
      sub_86A080(a1);
      a1 = v8;
    }
  }
}

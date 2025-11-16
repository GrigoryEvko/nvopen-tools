// Function: sub_8AEF00
// Address: 0x8aef00
//
__int64 __fastcall sub_8AEF00(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // r12
  unsigned int v6; // r14d
  int v7; // r15d
  int v8; // edx
  char v9; // si
  int v10; // ecx
  _QWORD *v11; // rax
  char v12; // al
  _QWORD *v13; // rax
  unsigned __int8 v15; // di
  char v16; // al
  _QWORD *v17; // rax

  if ( a3 )
  {
    v5 = (__int64)a3;
    v6 = 0;
    v7 = 0;
    do
    {
      while ( 1 )
      {
        v11 = (_QWORD *)*a2;
        if ( *a2 )
          break;
        v12 = *(_BYTE *)(v5 + 56);
        if ( (v12 & 0x10) != 0 )
        {
          if ( v7 )
            return 1;
LABEL_14:
          v13 = sub_725090(3u);
          *v13 = *a2;
          *a2 = v13;
          a2 = v13;
          v11 = (_QWORD *)*v13;
          if ( !v11 )
            return 1;
          goto LABEL_15;
        }
        if ( v6 || (v12 & 1) == 0 )
          return v6;
        v15 = 0;
        v16 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 80LL);
        if ( v16 != 3 )
          v15 = (v16 != 2) + 1;
        v17 = sub_725090(v15);
        *a2 = v17;
        sub_8AEEA0(a1, (__int64)v17, v5, a3);
        v11 = (_QWORD *)*a2;
        v5 = *(_QWORD *)v5;
        if ( !*a2 )
          return 1;
LABEL_9:
        if ( v7 )
          goto LABEL_15;
        a2 = (_QWORD *)*a2;
        if ( !v5 )
          return 1;
      }
      v8 = *((unsigned __int8 *)v11 + 8);
      if ( (_BYTE)v8 == 3 )
      {
        v6 = 1;
        if ( (*(_BYTE *)(v5 + 56) & 0x10) == 0 )
          goto LABEL_8;
      }
      else
      {
        v9 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 80LL);
        v10 = 0;
        if ( v9 != 3 )
          v10 = (v9 != 2) + 1;
        if ( v8 != v10 )
          return 0;
        if ( (*(_BYTE *)(v5 + 56) & 0x10) == 0 )
        {
LABEL_8:
          v5 = *(_QWORD *)v5;
          goto LABEL_9;
        }
      }
      if ( !v7 )
        goto LABEL_14;
LABEL_15:
      *((_BYTE *)v11 + 24) |= 8u;
      v7 = 1;
      a2 = (_QWORD *)*a2;
    }
    while ( v5 );
  }
  return 1;
}

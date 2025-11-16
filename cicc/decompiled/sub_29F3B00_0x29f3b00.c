// Function: sub_29F3B00
// Address: 0x29f3b00
//
__int64 __fastcall sub_29F3B00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rbx
  bool v7; // al
  _QWORD *v8; // rdi
  char v9; // cl
  char v10; // al
  __int16 v11; // dx
  char v12; // al
  __int64 v14; // rax

  v4 = a1 + 48;
  v5 = a2;
  if ( a2 != a1 + 48 )
  {
    v6 = a2;
    do
    {
      while ( 1 )
      {
        if ( !v6 )
          BUG();
        v12 = *(_BYTE *)(v6 - 24);
        v8 = (_QWORD *)(v6 - 24);
        if ( v12 != 60 )
          break;
        v7 = sub_B4D040((__int64)v8);
        v8 = (_QWORD *)(v6 - 24);
        if ( v7 )
          goto LABEL_4;
LABEL_10:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v4 == v6 )
          return v5;
      }
      if ( v12 != 85 )
        goto LABEL_10;
      v14 = *(_QWORD *)(v6 - 56);
      if ( !v14
        || *(_BYTE *)v14
        || *(_QWORD *)(v14 + 24) != *(_QWORD *)(v6 + 56)
        || (*(_BYTE *)(v14 + 33) & 0x20) == 0
        || *(_DWORD *)(v14 + 36) != 216 )
      {
        goto LABEL_10;
      }
LABEL_4:
      if ( v5 == v6 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        v10 = 0;
        v9 = 0;
        v5 = v6;
      }
      else
      {
        sub_B444E0(v8, v5, a3);
        v6 = *(_QWORD *)(v6 + 8);
        v9 = a3;
        v10 = BYTE1(a3);
      }
      LOBYTE(v11) = v9;
      HIBYTE(v11) = v10;
      LOWORD(a3) = v11;
    }
    while ( v4 != v6 );
  }
  return v5;
}

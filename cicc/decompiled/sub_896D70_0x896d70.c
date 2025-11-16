// Function: sub_896D70
// Address: 0x896d70
//
_QWORD *__fastcall sub_896D70(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r14
  bool v4; // r12
  _QWORD *v5; // r13
  _QWORD *v6; // rbx
  char v7; // al
  char v8; // al
  char v9; // al
  _QWORD *v10; // r15
  _QWORD *v11; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v3 = a2;
    v4 = a3 != 0;
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      v10 = v5;
      if ( (*(_BYTE *)(v3 + 56) & 0x10) == 0 )
        goto LABEL_3;
      v11 = sub_725090(3u);
      if ( !v6 )
        v6 = v11;
      v10 = v11;
      if ( v5 )
      {
        *v5 = v11;
        v7 = *(_BYTE *)(*(_QWORD *)(v3 + 8) + 80LL);
        if ( v7 == 3 )
        {
LABEL_19:
          v13 = *(_QWORD *)(v3 + 64);
          v5 = sub_725090(0);
          v5[4] = v13;
          if ( a1 )
            *(_QWORD *)(*(_QWORD *)(v13 + 168) + 16LL) = a1;
          goto LABEL_6;
        }
      }
      else
      {
LABEL_3:
        v7 = *(_BYTE *)(*(_QWORD *)(v3 + 8) + 80LL);
        if ( v7 == 3 )
          goto LABEL_19;
      }
      if ( v7 == 2 )
      {
        v5 = sub_725090(1u);
        v5[4] = *(_QWORD *)(v3 + 64);
      }
      else
      {
        v5 = sub_725090(2u);
        v5[4] = *(_QWORD *)(*(_QWORD *)(v3 + 64) + 104LL);
      }
LABEL_6:
      v8 = *(_BYTE *)(v3 + 56) & 0x10 | v5[3] & 0xEF;
      *((_BYTE *)v5 + 24) = v8;
      v9 = (8 * ((*(_BYTE *)(v3 + 56) & 0x50) != 0)) | v8 & 0xF7;
      *((_BYTE *)v5 + 24) = v9;
      if ( (v9 & 0x10) != 0 && v4 )
        sub_866610(v3, (__int64)v5);
      if ( !v6 )
        v6 = v5;
      if ( v10 )
        *v10 = v5;
      v3 = *(_QWORD *)v3;
      if ( !v3 )
        return v6;
    }
  }
  return 0;
}

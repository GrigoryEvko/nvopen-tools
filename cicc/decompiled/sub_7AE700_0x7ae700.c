// Function: sub_7AE700
// Address: 0x7ae700
//
void __fastcall sub_7AE700(__int64 a1, int a2, unsigned int a3, int a4, __int64 a5)
{
  __int64 v6; // r12
  char v7; // r14
  _QWORD *v8; // r15
  char v9; // al
  __int64 v10; // rbx
  bool v11; // [rsp+Fh] [rbp-31h]

  v6 = *(_QWORD *)(a1 + 8);
  v7 = a4 != 0;
  if ( a2 )
  {
    if ( v6 )
    {
      v8 = *(_QWORD **)(a1 + 8);
      do
      {
        if ( *(_DWORD *)(v6 + 28) == a2 )
        {
          if ( !a3 )
          {
            v11 = 0;
            goto LABEL_10;
          }
LABEL_26:
          while ( *(_DWORD *)(v6 + 32) < a3 && *(_WORD *)(v6 + 24) != 9 )
          {
            v6 = *(_QWORD *)v6;
            if ( !v6 )
            {
              v11 = 0;
              v7 = 0;
              goto LABEL_10;
            }
          }
          v11 = *(_DWORD *)(v6 + 28) == a3 - 1;
          if ( v8 != (_QWORD *)v6 )
            goto LABEL_31;
          goto LABEL_11;
        }
        v9 = *(_BYTE *)(v6 + 26);
        v6 = *(_QWORD *)v6;
        if ( v9 != 3 )
          v8 = (_QWORD *)v6;
      }
      while ( v6 );
      if ( !a3 )
        goto LABEL_37;
      if ( v8 )
      {
        v11 = 0;
        v7 = 0;
        goto LABEL_31;
      }
    }
  }
  else if ( a3 )
  {
    if ( v6 )
    {
      v8 = *(_QWORD **)(a1 + 8);
      goto LABEL_26;
    }
  }
  else
  {
    v8 = *(_QWORD **)(a1 + 8);
LABEL_37:
    v11 = 0;
    v7 &= v6 != 0;
LABEL_10:
    if ( v8 == (_QWORD *)v6 )
    {
LABEL_11:
      v10 = 0;
    }
    else
    {
      do
      {
LABEL_31:
        v10 = qword_4F08558;
        if ( qword_4F08558 )
          qword_4F08558 = *(_QWORD *)qword_4F08558;
        else
          v10 = sub_823970(112);
        *(_QWORD *)(v10 + 40) = 0;
        *(_QWORD *)v10 = 0;
        *(_BYTE *)(v10 + 26) = 0;
        *(_WORD *)(v10 + 24) = 0;
        *(_QWORD *)(v10 + 28) = 0;
        sub_7AE020((__int64)v8, (__m128i *)v10);
        if ( *(_QWORD *)(a5 + 8) )
          **(_QWORD **)(a5 + 16) = v10;
        else
          *(_QWORD *)(a5 + 8) = v10;
        *(_QWORD *)(a5 + 16) = v10;
        v8 = (_QWORD *)*v8;
      }
      while ( v8 != (_QWORD *)v6 );
    }
    if ( v7 )
    {
      v10 = qword_4F08558;
      if ( qword_4F08558 )
        qword_4F08558 = *(_QWORD *)qword_4F08558;
      else
        v10 = sub_823970(112);
      *(_QWORD *)(v10 + 40) = 0;
      *(_QWORD *)v10 = 0;
      *(_BYTE *)(v10 + 26) = 0;
      *(_WORD *)(v10 + 24) = 0;
      *(_QWORD *)(v10 + 28) = 0;
      sub_7AE020(v6, (__m128i *)v10);
      if ( *(_QWORD *)(a5 + 8) )
        **(_QWORD **)(a5 + 16) = v10;
      else
        *(_QWORD *)(a5 + 8) = v10;
      *(_QWORD *)(a5 + 16) = v10;
    }
    if ( v11 )
      *(_WORD *)(v10 + 24) = 44;
  }
}

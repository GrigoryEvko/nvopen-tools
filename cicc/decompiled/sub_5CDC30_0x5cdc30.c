// Function: sub_5CDC30
// Address: 0x5cdc30
//
__int64 __fastcall sub_5CDC30(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rdx
  char v7; // cl
  _QWORD *v8; // r12
  _QWORD *i; // r14
  __int64 v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rsi

  result = (__int64)&unk_4F074A0;
  if ( unk_4F074B0 )
  {
    result = sub_8D97B0(*(_QWORD *)(a1 + 288));
    if ( (_DWORD)result )
      return result;
  }
  v3 = *(_QWORD *)(a1 + 280);
  if ( *(_BYTE *)(v3 + 140) != 7 )
  {
    if ( (*(_BYTE *)(a1 + 125) & 0x10) == 0 )
    {
      do
LABEL_5:
        v3 = sub_8D48B0(v3, 0);
      while ( *(_BYTE *)(v3 + 140) != 7 );
LABEL_6:
      result = *(_QWORD *)(v3 + 168);
      v4 = *(_QWORD **)result;
      if ( *(_QWORD *)result )
      {
        do
        {
          v5 = v4[8];
          if ( v5 )
          {
            result = sub_736C60(5, v5);
            if ( result )
              result = sub_5CCAE0(8u, result);
          }
          v4 = (_QWORD *)*v4;
        }
        while ( v4 );
      }
      return result;
    }
LABEL_4:
    v3 = *(_QWORD *)(a1 + 288);
    if ( *(_BYTE *)(v3 + 140) == 7 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( *(_BYTE *)(a1 + 269) == 4 )
  {
    if ( (*(_BYTE *)(a1 + 125) & 0x10) == 0 )
      goto LABEL_6;
    goto LABEL_4;
  }
  if ( (*(_BYTE *)(a1 + 127) & 0x10) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 296);
    if ( v6 )
    {
      result = *(unsigned __int8 *)(v6 + 140);
      if ( (_BYTE)result == 12 )
      {
        result = *(_QWORD *)(a1 + 296);
        do
        {
          result = *(_QWORD *)(result + 160);
          v7 = *(_BYTE *)(result + 140);
        }
        while ( v7 == 12 );
        if ( !v7 )
          return result;
        do
          v6 = *(_QWORD *)(v6 + 160);
        while ( *(_BYTE *)(v6 + 140) == 12 );
      }
      else if ( !(_BYTE)result )
      {
        return result;
      }
      v8 = **(_QWORD ***)(v3 + 168);
      result = *(_QWORD *)(v6 + 168);
      for ( i = *(_QWORD **)result; v8; i = (_QWORD *)*i )
      {
        v10 = v8[8];
        if ( v10 )
        {
          result = sub_736C60(5, v10);
          v11 = result;
          if ( result )
          {
            v12 = i[8];
            if ( !v12 || (result = sub_736C60(5, v12)) == 0 )
              result = sub_6854C0(1855, v11 + 56, *(_QWORD *)a1);
          }
        }
        v8 = (_QWORD *)*v8;
      }
    }
  }
  return result;
}

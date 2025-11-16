// Function: sub_74B930
// Address: 0x74b930
//
__int64 __fastcall sub_74B930(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v3; // bl
  __int64 result; // rax
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax

  v2 = a1;
  if ( *(_BYTE *)(a2 + 153) )
  {
    while ( *(_BYTE *)(v2 + 140) == 12 )
    {
      if ( !*(_QWORD *)(v2 + 8) )
        break;
      v5 = v2;
      do
      {
        v5 = *(_QWORD *)(v5 + 160);
        v6 = *(_BYTE *)(v5 + 140);
      }
      while ( v6 == 12 );
      if ( v6 == 21 )
        break;
      v7 = v2;
      do
      {
        v7 = *(_QWORD *)(v7 + 160);
        v8 = *(_BYTE *)(v7 + 140);
      }
      while ( v8 == 12 );
      if ( !v8 )
        break;
      v9 = *(_QWORD *)(v2 + 40);
      if ( v9 )
      {
        if ( *(_BYTE *)(v9 + 28) == 3 && **(_QWORD ***)(v9 + 32) == qword_4D049B8 )
          break;
      }
      v2 = *(_QWORD *)(v2 + 160);
    }
    goto LABEL_3;
  }
  if ( a1 )
  {
LABEL_3:
    v3 = *(_BYTE *)(a2 + 161);
    *(_BYTE *)(a2 + 161) = 1;
    sub_74A390(v2, 0, 0, 0, 0, a2);
    result = sub_74D110(v2, 0, 0, a2);
    *(_BYTE *)(a2 + 161) = v3;
    return result;
  }
  return (*(__int64 (__fastcall **)(const char *))a2)("<null-type>");
}

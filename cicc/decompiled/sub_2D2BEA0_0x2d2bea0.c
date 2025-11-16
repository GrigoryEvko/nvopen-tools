// Function: sub_2D2BEA0
// Address: 0x2d2bea0
//
__int64 __fastcall sub_2D2BEA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // rcx
  __int64 v4; // rax
  char v5; // dl
  __int64 result; // rax
  bool v7; // zf
  char v8; // al
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v3 = *(_QWORD *)a2;
    while ( v3 >= *(_QWORD *)(v2 + 32) )
    {
      if ( v3 != *(_QWORD *)(v2 + 32) )
        goto LABEL_4;
      v8 = *(_BYTE *)(a2 + 24);
      if ( *(_BYTE *)(v2 + 56) )
      {
        if ( !v8 )
          break;
        v9 = *(_QWORD *)(a2 + 8);
        v10 = *(_QWORD *)(v2 + 40);
        if ( v9 < v10 || v9 == v10 && *(_QWORD *)(a2 + 16) < *(_QWORD *)(v2 + 48) )
          break;
        if ( v9 > v10 || *(_QWORD *)(v2 + 48) < *(_QWORD *)(a2 + 16) || *(_QWORD *)(a2 + 32) >= *(_QWORD *)(v2 + 64) )
          goto LABEL_4;
      }
      else if ( v8 || *(_QWORD *)(a2 + 32) >= *(_QWORD *)(v2 + 64) )
      {
LABEL_4:
        v4 = *(_QWORD *)(v2 + 24);
        v5 = 0;
        if ( !v4 )
          goto LABEL_9;
        goto LABEL_5;
      }
      v4 = *(_QWORD *)(v2 + 16);
      v5 = 1;
LABEL_8:
      if ( !v4 )
      {
LABEL_9:
        if ( v5 )
          goto LABEL_10;
LABEL_12:
        v7 = sub_2A4D650(v2 + 32, a2) == 0;
        result = v2;
        if ( !v7 )
          return 0;
        return result;
      }
LABEL_5:
      v2 = v4;
    }
    v4 = *(_QWORD *)(v2 + 16);
    v5 = 1;
    goto LABEL_8;
  }
  v2 = a1 + 8;
LABEL_10:
  result = 0;
  if ( *(_QWORD *)(a1 + 24) != v2 )
  {
    v2 = sub_220EF80(v2);
    goto LABEL_12;
  }
  return result;
}

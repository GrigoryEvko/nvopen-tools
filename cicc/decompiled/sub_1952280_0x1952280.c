// Function: sub_1952280
// Address: 0x1952280
//
__int64 __fastcall sub_1952280(__int64 a1)
{
  unsigned __int64 v1; // r14
  __int64 v2; // rbx
  int v3; // r12d
  unsigned int v4; // r13d
  unsigned int i; // r12d
  __int64 v6; // r15
  int v7; // ebx
  unsigned int v8; // eax
  unsigned int v10; // [rsp+8h] [rbp-38h]
  int v11; // [rsp+Ch] [rbp-34h]

  v1 = sub_157EBA0(a1);
  v2 = *(_QWORD *)(sub_15F4DF0(v1, 0) + 8);
  if ( v2 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v2) + 16) - 25) > 9u )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        goto LABEL_23;
    }
    v3 = 0;
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v2) + 16) - 25) <= 9u )
      {
        v2 = *(_QWORD *)(v2 + 8);
        ++v3;
        if ( !v2 )
          goto LABEL_7;
      }
    }
LABEL_7:
    v4 = v3 + 1;
    v11 = sub_15F4D60(v1);
    if ( v11 != 1 )
      goto LABEL_8;
    return 0;
  }
LABEL_23:
  v4 = 0;
  v11 = sub_15F4D60(v1);
  if ( v11 == 1 )
    return 0;
LABEL_8:
  v10 = 0;
  for ( i = 1; i != v11; ++i )
  {
    v6 = *(_QWORD *)(sub_15F4DF0(v1, i) + 8);
    if ( v6 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) > 9u )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_21;
      }
      v7 = 0;
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) <= 9u )
        {
          v6 = *(_QWORD *)(v6 + 8);
          ++v7;
          if ( !v6 )
            goto LABEL_15;
        }
      }
LABEL_15:
      v8 = v7 + 1;
    }
    else
    {
LABEL_21:
      v8 = 0;
    }
    if ( v8 < v4 )
    {
      v10 = i;
      v4 = v8;
    }
  }
  return v10;
}

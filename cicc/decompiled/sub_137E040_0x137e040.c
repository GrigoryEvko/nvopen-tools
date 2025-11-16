// Function: sub_137E040
// Address: 0x137e040
//
__int64 __fastcall sub_137E040(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v10; // rax

  if ( (unsigned int)sub_15F4D60(a1) != 1 )
  {
    v4 = *(_QWORD *)(sub_15F4DF0(a1, a2) + 8);
    do
    {
      if ( !v4 )
        BUG();
      v5 = sub_1648700(v4);
      v4 = *(_QWORD *)(v4 + 8);
      v6 = v5;
    }
    while ( (unsigned __int8)(*(_BYTE *)(v5 + 16) - 25) > 9u );
    while ( v4 )
    {
      v7 = sub_1648700(v4);
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 25) <= 9u )
      {
        if ( !(_BYTE)a3 )
          return 1;
        v8 = *(_QWORD *)(v6 + 40);
        if ( v8 != *(_QWORD *)(v7 + 40) )
          return a3;
        while ( 1 )
        {
          v4 = *(_QWORD *)(v4 + 8);
          if ( !v4 )
            break;
          v10 = sub_1648700(v4);
          if ( (unsigned __int8)(*(_BYTE *)(v10 + 16) - 25) <= 9u && v8 != *(_QWORD *)(v10 + 40) )
            return a3;
        }
        return 0;
      }
      v4 = *(_QWORD *)(v4 + 8);
    }
  }
  return 0;
}

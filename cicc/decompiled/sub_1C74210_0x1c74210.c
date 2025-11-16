// Function: sub_1C74210
// Address: 0x1c74210
//
_QWORD *__fastcall sub_1C74210(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r13
  _QWORD *v8; // rax
  __int64 v9; // rdx

  v3 = 0;
  v5 = sub_1C741B0(a1, a2);
  if ( *(_BYTE *)(v5 + 16) > 0x17u )
  {
    v3 = *(_QWORD *)(v5 + 8);
    v7 = v6;
    if ( v3 )
    {
      while ( 1 )
      {
        v8 = sub_1648700(v3);
        if ( *((_BYTE *)v8 + 16) == 77 )
        {
          v9 = v8[5];
          if ( v7 == v9 )
            break;
          if ( a3 == v9 )
          {
            v8 = sub_1648700(v8[1]);
            if ( *((_BYTE *)v8 + 16) == 77 && v7 == v8[5] )
              break;
          }
        }
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return (_QWORD *)v3;
      }
      return v8;
    }
  }
  return (_QWORD *)v3;
}

// Function: sub_690A60
// Address: 0x690a60
//
__int64 __fastcall sub_690A60(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // [rsp+0h] [rbp-30h] BYREF
  _QWORD *v10; // [rsp+8h] [rbp-28h] BYREF

  v9 = 0;
  v10 = 0;
  if ( unk_4D03C50 )
  {
    v4 = *(_QWORD **)(unk_4D03C50 + 136LL);
    if ( v4 )
    {
      if ( *v4 )
      {
        do
        {
          v6 = (_QWORD *)sub_6E1C80(v4);
          if ( v9 )
          {
            *v10 = v6;
            v10 = v6;
            v5 = unk_4D03C50;
            if ( !unk_4D03C50 )
              return (__int64)v9;
          }
          else
          {
            v9 = v6;
            v10 = v6;
            v5 = unk_4D03C50;
            if ( !unk_4D03C50 )
              return (__int64)v9;
          }
          v4 = *(_QWORD **)(v5 + 136);
        }
        while ( v4 && *v4 );
        return (__int64)v9;
      }
    }
  }
  if ( a1 )
  {
    do
    {
      if ( (*(_BYTE *)(a1 + 25) & 0x10) != 0 )
        break;
      if ( (*(_BYTE *)(a1 + 26) & 4) != 0 )
      {
        sub_68A480(a1, (__int64 *)&v9, &v10, a2);
      }
      else
      {
        v8 = (_QWORD *)sub_6F8A60(a1, a2, a3);
        if ( v9 )
        {
          a3 = v10;
          *v10 = v8;
        }
        else
        {
          v9 = v8;
        }
        v10 = v8;
      }
      a1 = *(_QWORD *)(a1 + 16);
    }
    while ( a1 );
    return (__int64)v9;
  }
  return 0;
}

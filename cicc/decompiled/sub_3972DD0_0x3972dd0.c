// Function: sub_3972DD0
// Address: 0x3972dd0
//
char __fastcall sub_3972DD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 i; // r14
  char v6; // al
  __int64 v7; // rcx
  int v8; // r13d
  _QWORD *v9; // rdi
  __int64 v11; // [rsp+8h] [rbp-48h]
  unsigned __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_396DD80(a1);
  if ( *(_BYTE *)(v3 + 776) )
  {
    v4 = *(_QWORD *)(a2 + 16);
    for ( i = a2 + 8; i != v4; v4 = *(_QWORD *)(v4 + 8) )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        LOBYTE(v3) = *(_BYTE *)(v4 - 24) >> 6;
        if ( (_BYTE)v3 == 2 )
        {
          LOBYTE(v3) = sub_15E4F60(v4 - 56);
          if ( !(_BYTE)v3 && (*(_BYTE *)(v4 + 24) & 1) != 0 )
          {
            v6 = *(_BYTE *)(v4 - 24) & 0xF;
            if ( ((v6 + 9) & 0xFu) <= 1 || (LOBYTE(v3) = (v6 + 15) & 0xF, (unsigned __int8)v3 <= 2u) )
            {
              v3 = *(_QWORD *)(v4 - 80);
              if ( !v3 )
                BUG();
              if ( *(_BYTE *)(v3 + 16) <= 3u )
              {
                v7 = *(_QWORD *)(v4 - 48);
                v8 = 0;
                if ( v7 )
                {
                  do
                  {
                    v11 = v7;
                    v9 = sub_1648700(v7);
                    if ( *((_BYTE *)v9 + 16) >= 0x11u )
                      v9 = 0;
                    LODWORD(v3) = sub_396B840((__int64)v9);
                    v8 += v3;
                    v7 = *(_QWORD *)(v11 + 8);
                  }
                  while ( v7 );
                  if ( v8 )
                    break;
                }
              }
            }
          }
        }
        v4 = *(_QWORD *)(v4 + 8);
        if ( i == v4 )
          return v3;
      }
      v12[0] = sub_396EAF0(a1, v4 - 56);
      v3 = sub_3972B10(a1 + 320, v12);
      *(_QWORD *)v3 = v4 - 56;
      *(_DWORD *)(v3 + 8) = v8;
    }
  }
  return v3;
}

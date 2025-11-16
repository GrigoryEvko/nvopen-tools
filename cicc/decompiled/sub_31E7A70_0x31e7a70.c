// Function: sub_31E7A70
// Address: 0x31e7a70
//
char __fastcall sub_31E7A70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  char v5; // al
  __int64 v6; // r9
  int v7; // r13d
  __int64 v8; // rcx
  __int64 v9; // r8
  _BYTE *v10; // rdi
  __int64 v11; // r8
  int v12; // r10d
  int v13; // r10d
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 i; // [rsp+8h] [rbp-48h]
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_31DA6B0(a1);
  if ( *(_BYTE *)(v3 + 936) )
  {
    v4 = *(_QWORD *)(a2 + 16);
    LOBYTE(v3) = a2 + 8;
    for ( i = a2 + 8; i != v4; v4 = *(_QWORD *)(v4 + 8) )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        LOBYTE(v3) = *(_BYTE *)(v4 - 24) >> 6;
        if ( (_BYTE)v3 == 2 )
        {
          LOBYTE(v3) = sub_B2FC80(v4 - 56);
          if ( !(_BYTE)v3 && (*(_BYTE *)(v4 + 24) & 1) != 0 )
          {
            v5 = *(_BYTE *)(v4 - 24) & 0xF;
            if ( ((v5 + 15) & 0xFu) <= 2 || (LOBYTE(v3) = (v5 + 9) & 0xF, (unsigned __int8)v3 <= 1u) )
            {
              v3 = *(_QWORD *)(v4 - 88);
              if ( *(_BYTE *)v3 <= 3u )
              {
                v6 = *(_QWORD *)(v4 - 40);
                if ( v6 )
                {
                  v7 = 0;
                  do
                  {
                    v8 = *(_QWORD *)(v6 + 24);
                    LOBYTE(v3) = *(_BYTE *)v8;
                    if ( *(_BYTE *)v8 <= 0x15u )
                    {
                      if ( (_BYTE)v3 == 3 )
                      {
                        ++v7;
                      }
                      else
                      {
                        v9 = *(_QWORD *)(v8 + 16);
                        if ( v9 )
                        {
                          do
                          {
                            v10 = *(_BYTE **)(v9 + 24);
                            if ( *v10 >= 0x16u )
                              v10 = 0;
                            LODWORD(v3) = sub_31D63F0((__int64)v10);
                            v9 = *(_QWORD *)(v11 + 8);
                            v13 = v3 + v12;
                          }
                          while ( v9 );
                          v7 += v13;
                        }
                      }
                    }
                    v6 = *(_QWORD *)(v6 + 8);
                  }
                  while ( v6 );
                  if ( v7 )
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
      v20[0] = sub_31DB510(a1, v4 - 56);
      v3 = sub_31E7750(a1 + 352, v20, v14, v15, v16, v17);
      *(_QWORD *)v3 = v4 - 56;
      *(_DWORD *)(v3 + 8) = v7;
    }
  }
  return v3;
}

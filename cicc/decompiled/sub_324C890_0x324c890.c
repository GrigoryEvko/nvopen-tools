// Function: sub_324C890
// Address: 0x324c890
//
void __fastcall sub_324C890(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  bool v4; // dl
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // [rsp+10h] [rbp-40h]

  if ( a3 )
  {
    v3 = *(_BYTE *)(a3 - 16);
    v4 = (v3 & 2) != 0;
    v5 = (v3 & 2) != 0 ? *(unsigned int *)(a3 - 24) : (*(_WORD *)(a3 - 16) >> 6) & 0xFu;
    if ( (unsigned int)v5 > 1 )
    {
      v6 = 8;
      v10 = 8 * v5;
      while ( 1 )
      {
        if ( v4 )
        {
          v7 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + v6);
          v8 = a2;
          if ( !v7 )
            goto LABEL_13;
        }
        else
        {
          v7 = *(_QWORD *)(a3 - 16 - 8LL * ((v3 >> 2) & 0xF) + v6);
          v8 = a2;
          if ( !v7 )
          {
LABEL_13:
            v6 += 8;
            sub_324C6D0(a1, 24, v8, 0);
            if ( v6 == v10 )
              return;
            goto LABEL_10;
          }
        }
        v9 = sub_324C6D0(a1, 5, v8, 0);
        sub_32495E0(a1, v9, v7, 73);
        if ( (*(_BYTE *)(v7 + 20) & 0x40) != 0 )
          sub_3249FA0(a1, v9, 52);
        v6 += 8;
        if ( v6 == v10 )
          return;
LABEL_10:
        v3 = *(_BYTE *)(a3 - 16);
        v4 = (v3 & 2) != 0;
      }
    }
  }
}

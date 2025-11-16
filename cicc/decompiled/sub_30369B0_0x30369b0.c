// Function: sub_30369B0
// Address: 0x30369b0
//
__int64 __fastcall sub_30369B0(unsigned __int16 *a1)
{
  __int64 v1; // rbp
  int v2; // eax
  __int64 v3; // rax
  char v5; // cl
  __int64 v6; // rax
  int v7; // eax
  __int16 v8; // di
  int v9; // edx
  int v10; // esi
  __int64 v12; // [rsp-28h] [rbp-28h] BYREF
  char v13; // [rsp-20h] [rbp-20h]
  __int64 v14; // [rsp-8h] [rbp-8h]

  v2 = (unsigned __int16)word_4456580[*a1 - 1];
  if ( (unsigned __int16)v2 <= 1u || (unsigned __int16)(v2 - 504) <= 7u )
    BUG();
  v14 = v1;
  v3 = 16LL * (v2 - 1);
  v5 = byte_444C4A0[v3 + 8];
  v6 = *(_QWORD *)&byte_444C4A0[v3];
  v13 = v5;
  v12 = v6;
  v7 = sub_CA1930(&v12);
  v8 = 2;
  if ( v7 != 1 )
  {
    v8 = 3;
    if ( v7 != 2 )
    {
      v8 = 4;
      if ( v7 != 4 )
      {
        v8 = 5;
        if ( v7 != 8 )
        {
          v8 = 6;
          if ( v7 != 16 )
          {
            v8 = 7;
            if ( v7 != 32 )
            {
              v8 = 8;
              if ( v7 != 64 )
                v8 = 9 * (v7 == 128);
            }
          }
        }
      }
    }
  }
  v9 = *a1;
  v10 = word_4456340[v9 - 1];
  if ( (unsigned __int16)(v9 - 176) > 0x34u )
    return sub_2D43050(v8, v10);
  else
    return sub_2D43AD0(v8, v10);
}

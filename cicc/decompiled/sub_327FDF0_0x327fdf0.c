// Function: sub_327FDF0
// Address: 0x327fdf0
//
__int64 __fastcall sub_327FDF0(unsigned __int16 *a1, __int64 a2)
{
  unsigned __int16 v2; // bx
  __int64 v3; // r12
  int v4; // eax
  __int64 v5; // rax
  char v6; // cl
  __int64 v7; // rax
  int v8; // eax
  __int16 v9; // di
  int v10; // esi
  __int64 v12; // [rsp+0h] [rbp-20h] BYREF
  char v13; // [rsp+8h] [rbp-18h]

  v2 = *a1;
  if ( !*a1 )
    return sub_300A990(a1, a2);
  v3 = v2 - 1;
  v4 = (unsigned __int16)word_4456580[v3];
  if ( (unsigned __int16)v4 <= 1u || (unsigned __int16)(word_4456580[v3] - 504) <= 7u )
    BUG();
  v5 = 16LL * (v4 - 1);
  v6 = byte_444C4A0[v5 + 8];
  v7 = *(_QWORD *)&byte_444C4A0[v5];
  v13 = v6;
  v12 = v7;
  v8 = sub_CA1930(&v12);
  v9 = 2;
  if ( v8 != 1 )
  {
    v9 = 3;
    if ( v8 != 2 )
    {
      v9 = 4;
      if ( v8 != 4 )
      {
        v9 = 5;
        if ( v8 != 8 )
        {
          v9 = 6;
          if ( v8 != 16 )
          {
            v9 = 7;
            if ( v8 != 32 )
            {
              v9 = 8;
              if ( v8 != 64 )
                v9 = 9 * (v8 == 128);
            }
          }
        }
      }
    }
  }
  v10 = word_4456340[v3];
  if ( (unsigned __int16)(v2 - 176) <= 0x34u )
    return (unsigned __int16)sub_2D43AD0(v9, v10);
  else
    return (unsigned __int16)sub_2D43050(v9, v10);
}

// Function: sub_2535850
// Address: 0x2535850
//
void __fastcall sub_2535850(__int64 a1, __m128i *a2, __int64 a3, char a4)
{
  __int64 *v5; // r15
  __int64 *v6; // r12
  int v7; // eax
  unsigned __int64 v8; // r12
  __int64 *v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+8h] [rbp-48h]
  _BYTE v11[64]; // [rsp+10h] [rbp-40h] BYREF

  v10 = 0x200000000LL;
  v9 = (__int64 *)v11;
  sub_2515D00(a1, a2, dword_438A680, 3, (__int64)&v9, a4);
  v5 = v9;
  v6 = &v9[(unsigned int)v10];
  if ( v6 != v9 )
  {
    while ( 1 )
    {
      v7 = sub_A71AE0(v5);
      if ( v7 == 51 )
        break;
      if ( v7 == 78 )
      {
        *(_WORD *)(a3 + 8) |= 0x101u;
LABEL_4:
        if ( v6 == ++v5 )
          goto LABEL_9;
      }
      else
      {
        if ( v7 != 50 )
          BUG();
        ++v5;
        *(_WORD *)(a3 + 8) |= 0x303u;
        if ( v6 == v5 )
        {
LABEL_9:
          v5 = v9;
          goto LABEL_10;
        }
      }
    }
    *(_WORD *)(a3 + 8) |= 0x202u;
    goto LABEL_4;
  }
LABEL_10:
  v8 = a2->m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
  if ( (a2->m128i_i64[0] & 3) == 3 )
    v8 = *(_QWORD *)(v8 + 24);
  if ( *(_BYTE *)v8 > 0x1Cu )
  {
    if ( !(unsigned __int8)sub_B46420(v8) )
      *(_WORD *)(a3 + 8) |= 0x101u;
    if ( !(unsigned __int8)sub_B46490(v8) )
      *(_WORD *)(a3 + 8) |= 0x202u;
  }
  if ( v5 != (__int64 *)v11 )
    _libc_free((unsigned __int64)v5);
}

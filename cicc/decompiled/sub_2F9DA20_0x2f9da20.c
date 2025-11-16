// Function: sub_2F9DA20
// Address: 0x2f9da20
//
unsigned __int64 __fastcall sub_2F9DA20(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rsi
  __int16 v4; // ax
  __int16 v6; // bx
  __int16 v7; // r12
  unsigned __int64 v8; // rdi
  __int16 v9; // dx
  unsigned __int64 v10; // [rsp+10h] [rbp-20h] BYREF
  __int64 v11; // [rsp+18h] [rbp-18h]

  v10 = a1;
  v11 = a2;
  if ( a1 )
  {
    v3 = *(_QWORD *)a3;
    if ( *(_QWORD *)a3 )
    {
      v6 = v11;
      v7 = *(_WORD *)(a3 + 8);
      if ( a1 > 0xFFFFFFFF || v3 > 0xFFFFFFFF )
      {
        v8 = sub_F04140(a1, v3);
      }
      else
      {
        v8 = v3 * a1;
        v9 = 0;
      }
      v10 = v8;
      LOWORD(v11) = v9;
      sub_D78C90((__int64)&v10, (__int16)(v6 + v7));
    }
    else
    {
      v4 = *(_WORD *)(a3 + 8);
      v10 = 0;
      LOWORD(v11) = v4;
    }
  }
  return v10;
}

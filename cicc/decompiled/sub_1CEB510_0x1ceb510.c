// Function: sub_1CEB510
// Address: 0x1ceb510
//
void __fastcall sub_1CEB510(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  _QWORD *v3; // rax
  _BYTE *v4; // rsi
  _QWORD *v5; // [rsp+8h] [rbp-28h] BYREF

  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v3 = sub_1648700(i);
    if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 75) <= 1u )
    {
      v5 = v3;
      v4 = *(_BYTE **)(a1 + 192);
      if ( v4 == *(_BYTE **)(a1 + 200) )
      {
        sub_1CEB380(a1 + 184, v4, &v5);
      }
      else
      {
        if ( v4 )
        {
          *(_QWORD *)v4 = v3;
          v4 = *(_BYTE **)(a1 + 192);
        }
        *(_QWORD *)(a1 + 192) = v4 + 8;
      }
    }
  }
}

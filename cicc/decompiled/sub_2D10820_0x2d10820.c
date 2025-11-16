// Function: sub_2D10820
// Address: 0x2d10820
//
void __fastcall sub_2D10820(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r13
  _QWORD *v6; // r12
  char v7; // dh
  __int64 v8; // rsi
  char v9; // dl
  __int64 v10; // rax

  v3 = *(_QWORD *)(a2 + 80);
  v4 = *(_QWORD *)(a1 + 280);
  v5 = *(_QWORD *)(a1 + 272);
  if ( v3 )
    v3 -= 24;
  for ( ; v5 != v4; v4 -= 8 )
  {
    v6 = (_QWORD *)sub_2D10810(*(_QWORD *)(v4 - 8));
    if ( sub_B4D040((__int64)v6) )
    {
      LOBYTE(v2) = 1;
      v8 = sub_AA4FF0(v3);
      v9 = 0;
      if ( v8 )
        v9 = v7;
      v10 = v2;
      BYTE1(v10) = v9;
      sub_B444E0(v6, v8, v10);
    }
  }
}

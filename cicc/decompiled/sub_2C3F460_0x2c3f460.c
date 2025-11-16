// Function: sub_2C3F460
// Address: 0x2c3f460
//
void __fastcall sub_2C3F460(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
  {
    sub_2AC65C0(a1, (__int64 *)v3, (__int64 *)a2);
  }
  else
  {
    if ( v3 )
    {
      v4 = *(_QWORD *)a2;
      *(_BYTE *)(v3 + 24) = 0;
      *(_QWORD *)v3 = v4;
      if ( *(_BYTE *)(a2 + 24) )
      {
        *(_QWORD *)(v3 + 8) = *(_QWORD *)(a2 + 8);
        v5 = *(_QWORD *)(a2 + 16);
        *(_BYTE *)(v3 + 24) = 1;
        *(_QWORD *)(v3 + 16) = v5;
      }
      v3 = a1[1];
    }
    a1[1] = v3 + 32;
  }
}

// Function: sub_390D5F0
// Address: 0x390d5f0
//
void __fastcall sub_390D5F0(__int64 a1, __int64 a2, bool *a3)
{
  __int64 v3; // rbp
  char v4; // al
  _BYTE *v5; // r8
  _QWORD v6[2]; // [rsp-10h] [rbp-10h] BYREF

  v4 = *(_BYTE *)(a2 + 8) & 8;
  if ( a3 )
    *a3 = v4 == 0;
  if ( !v4 )
  {
    v6[1] = v3;
    *(_BYTE *)(a2 + 8) |= 8u;
    v5 = *(_BYTE **)(a1 + 64);
    v6[0] = a2;
    if ( v5 == *(_BYTE **)(a1 + 72) )
    {
      sub_390D460(a1 + 56, v5, v6);
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = a2;
        v5 = *(_BYTE **)(a1 + 64);
      }
      *(_QWORD *)(a1 + 64) = v5 + 8;
    }
  }
}

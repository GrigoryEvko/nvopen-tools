// Function: sub_30CA5F0
// Address: 0x30ca5f0
//
void __fastcall sub_30CA5F0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rsi
  bool v4; // cc
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = *(_BYTE *)(a1 + 136) == 0;
  *(_QWORD *)a1 = &unk_4A32518;
  if ( !v2 )
  {
    v2 = *(_BYTE *)(a1 + 128) == 0;
    *(_BYTE *)(a1 + 136) = 0;
    if ( !v2 )
    {
      v4 = *(_DWORD *)(a1 + 120) <= 0x40u;
      *(_BYTE *)(a1 + 128) = 0;
      if ( !v4 )
      {
        v5 = *(_QWORD *)(a1 + 112);
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      if ( *(_DWORD *)(a1 + 104) > 0x40u )
      {
        v6 = *(_QWORD *)(a1 + 96);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
    }
  }
  v3 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A1F3E0;
  if ( v3 )
    sub_B91220(a1 + 32, v3);
}

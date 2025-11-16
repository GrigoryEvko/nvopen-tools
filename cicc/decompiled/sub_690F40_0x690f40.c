// Function: sub_690F40
// Address: 0x690f40
//
void __fastcall sub_690F40(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r13
  __int64 v6; // rbx

  if ( *(_BYTE *)(a1 + 8) == 1 && (*(_BYTE *)(a1 + 24) & 1) == 0 )
  {
    v3 = sub_6E2EF0();
    v4 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 48) = v3;
    v5 = v3 + 8;
    sub_6E6A50(v4, v3 + 8);
    if ( unk_4F07734 )
    {
      sub_6E4E90(v5, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 152LL));
      if ( unk_4D03C50 )
      {
        if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 2) != 0 )
        {
          v6 = *(_QWORD *)(a1 + 32);
          *(_QWORD *)(v6 + 152) = sub_6E3700(v5, *(_QWORD *)(v6 + 152));
        }
      }
    }
  }
}

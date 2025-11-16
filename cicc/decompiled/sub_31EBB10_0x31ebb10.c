// Function: sub_31EBB10
// Address: 0x31ebb10
//
void __fastcall sub_31EBB10(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax

  v2 = *a2;
  if ( (*(_BYTE *)(*a2 + 7) & 0x20) != 0 )
  {
    v3 = sub_B91C10(v2, 36);
    if ( v3 )
    {
      v4 = *(_BYTE *)(v3 - 16);
      if ( (v4 & 2) != 0 )
        v5 = *(_QWORD *)(v3 - 32);
      else
        v5 = v3 - 8LL * ((v4 >> 2) & 0xF) - 16;
      v6 = *(_QWORD *)(*(_QWORD *)v5 + 136LL);
      v7 = sub_B2BEC0(v2);
      sub_31EA6F0(a1, v7, v6, 0);
    }
  }
}

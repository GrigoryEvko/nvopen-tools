// Function: sub_30D1940
// Address: 0x30d1940
//
void __fastcall sub_30D1940(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  unsigned __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rdi

  if ( *(_BYTE *)(a1 + 714) )
  {
    v3 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 32))(*(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 72));
    if ( !sub_FDD2C0(v3, a2, 0) )
      *(_DWORD *)(a1 + 724) += *(_DWORD *)(a1 + 716) - *(_DWORD *)(a1 + 720);
  }
  v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == a2 + 48 )
  {
    v6 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = *(unsigned __int8 *)(v4 - 24);
    v6 = v4 - 24;
    if ( (unsigned int)(v5 - 30) >= 0xB )
      v6 = 0;
  }
  if ( *(_BYTE *)(a1 + 776) )
  {
    if ( (unsigned int)sub_B46E30(v6) > 1 )
    {
      *(_DWORD *)(a1 + 704) -= *(_DWORD *)(a1 + 660);
      *(_BYTE *)(a1 + 776) = 0;
    }
  }
}

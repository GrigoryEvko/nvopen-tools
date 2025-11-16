// Function: sub_2F5E800
// Address: 0x2f5e800
//
__int64 __fastcall sub_2F5E800(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = sub_22077B0(0x10u);
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = 3;
    *(_QWORD *)v2 = off_4A2B3F8;
  }
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return 0;
}

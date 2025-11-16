// Function: sub_3597BA0
// Address: 0x3597ba0
//
__int64 __fastcall sub_3597BA0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = sub_22077B0(0x18u);
  if ( v2 )
  {
    *(_DWORD *)(v2 + 8) = 1;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)v2 = &unk_4A39C00;
  }
  v3 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v2;
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  return 0;
}

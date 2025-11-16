// Function: sub_B59CA0
// Address: 0xb59ca0
//
__int64 __fastcall sub_B59CA0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v3; // rax
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( *(_DWORD *)(v1 + 36) == 199 )
    return *(_QWORD *)(a1 + 32 * (4LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v3 = (_QWORD *)sub_B43CA0(a1);
  v4 = sub_BCB2E0(*v3);
  return sub_ACD640(v4, 1, 0);
}

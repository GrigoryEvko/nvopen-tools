// Function: sub_ED2B90
// Address: 0xed2b90
//
char __fastcall sub_ED2B90(__int64 a1)
{
  _BYTE *v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rax

  v1 = sub_BA8CD0(a1, (__int64)"__llvm_profile_raw_version", 0x1Au, 1);
  if ( !v1 )
    goto LABEL_3;
  v2 = (__int64)v1;
  if ( (v1[32] & 0xFu) - 7 <= 1 )
    goto LABEL_3;
  LOBYTE(v3) = sub_B2FC80((__int64)v1);
  if ( (_BYTE)v3 )
    return v3;
  if ( !sub_B2FC80(v2) && (v4 = *(_QWORD *)(v2 - 32)) != 0 && *(_BYTE *)v4 == 17 )
  {
    v5 = *(_QWORD *)(v4 + 24);
    if ( *(_DWORD *)(v4 + 32) > 0x40u )
      v5 = *(_QWORD *)v5;
    return HIBYTE(v5) & 1;
  }
  else
  {
LABEL_3:
    LOBYTE(v3) = 0;
  }
  return v3;
}

// Function: sub_745B20
// Address: 0x745b20
//
__int64 __fastcall sub_745B20(_QWORD *a1, unsigned int *a2, int a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v9; // rdi
  unsigned int (__fastcall *v10)(__int64, _QWORD); // rax
  __int64 v11; // rax
  unsigned int v12; // r12d
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // [rsp+8h] [rbp-48h]
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *a1;
  if ( !(unsigned int)sub_8D57E0(*a1, v16) )
    return 0;
  v9 = v16[0];
  if ( (*(_BYTE *)(v16[0] + 89LL) & 1) != 0 )
  {
    if ( *(_BYTE *)(a4 + 139) )
      return 0;
  }
  if ( a3 && (*(_BYTE *)(v16[0] + 140LL) & 0xFB) == 8 )
  {
    v15 = v16[0];
    if ( (sub_8D4C10(v16[0], dword_4F077C4 != 2) & 1) != 0 )
      return 0;
    v9 = v15;
  }
  if ( *(_BYTE *)(a4 + 138) )
    return 0;
  v10 = *(unsigned int (__fastcall **)(__int64, _QWORD))(a4 + 96);
  if ( v10 )
  {
    if ( v10(v9, 0) )
      return 0;
  }
  v11 = sub_8D40F0(v16[0]);
  if ( *(_BYTE *)(v11 + 140) == 12 )
    v12 = ~(unsigned int)sub_8D4C10(v11, 1);
  else
    v12 = -1;
  v13 = sub_8D40F0(v7);
  v14 = 0;
  if ( *(_BYTE *)(v13 + 140) == 12 )
    v14 = sub_8D4C10(v13, 1);
  if ( a3 )
    v14 &= ~1u;
  *a2 = v12 & v14;
  *a1 = v16[0];
  return 1;
}

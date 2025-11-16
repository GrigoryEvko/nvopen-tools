// Function: sub_1039110
// Address: 0x1039110
//
__int64 __fastcall sub_1039110(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rax
  _BYTE *v3; // r12
  _DWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // r8d
  __int64 v8; // rdx

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 8LL * ((v1 >> 2) & 0xF) - 16;
  v3 = *(_BYTE **)(v2 + 8);
  if ( *v3 )
    v3 = 0;
  v4 = (_DWORD *)sub_B91420((__int64)v3);
  if ( v5 == 4 && *v4 == 1684828003 )
    return 2;
  v6 = sub_B91420((__int64)v3);
  v7 = 1;
  if ( v8 != 3 )
    return v7;
  if ( *(_WORD *)v6 == 28520 && (v7 = 4, *(_BYTE *)(v6 + 2) == 116) )
    return v7;
  else
    return 1;
}

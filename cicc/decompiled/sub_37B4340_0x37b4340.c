// Function: sub_37B4340
// Address: 0x37b4340
//
unsigned __int64 __fastcall sub_37B4340(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  unsigned __int64 v3; // r8
  _QWORD *v4; // rcx
  unsigned __int64 v5; // rax

  v2 = *(_QWORD **)(a2 + 40);
  v3 = 0;
  v4 = &v2[2 * *(unsigned int *)(a2 + 48)];
  if ( v4 == v2 )
    return v3;
  while ( 1 )
  {
    v5 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(v5 + 249) & 4) == 0 )
      break;
LABEL_6:
    v2 += 2;
    if ( v4 == v2 )
      return v3;
  }
  if ( !v3 || v5 == v3 )
  {
    v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_6;
  }
  return 0;
}

// Function: sub_2B2DA70
// Address: 0x2b2da70
//
__int64 __fastcall sub_2B2DA70(__int64 a1, _QWORD *a2, __int64 a3)
{
  int v3; // r14d
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  _BYTE *v6; // rax
  _BYTE **v8; // rax

  v3 = a3;
  v4 = &a2[a3];
  v5 = a2;
  if ( a2 == v4 )
    return 1;
  while ( 1 )
  {
    v6 = (_BYTE *)*v5;
    if ( *(_BYTE *)*v5 != 62 )
      break;
    v8 = (v6[7] & 0x40) != 0 ? (_BYTE **)*((_QWORD *)v6 - 1) : (_BYTE **)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
    if ( !*v8 || !(unsigned __int8)sub_2B09240(*v8, v3, *(_QWORD *)(a1 + 3296), 1) )
      break;
    if ( v4 == ++v5 )
      return 1;
  }
  return 0;
}

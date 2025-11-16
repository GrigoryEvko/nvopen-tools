// Function: sub_B9F8A0
// Address: 0xb9f8a0
//
__int64 __fastcall sub_B9F8A0(__int64 *a1, _BYTE *a2)
{
  _BYTE *v3; // rdx
  __int64 v4; // rdi
  int v5; // eax
  int v6; // esi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  _BYTE *v9; // r8
  int v11; // eax
  int v12; // r9d

  v3 = sub_B9F650(a1, a2);
  v4 = *(_QWORD *)(*a1 + 608);
  v5 = *(_DWORD *)(*a1 + 624);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v8 = (_QWORD *)(v4 + 16LL * v7);
    v9 = (_BYTE *)*v8;
    if ( v3 == (_BYTE *)*v8 )
      return v8[1];
    v11 = 1;
    while ( v9 != (_BYTE *)-4096LL )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (_QWORD *)(v4 + 16LL * v7);
      v9 = (_BYTE *)*v8;
      if ( v3 == (_BYTE *)*v8 )
        return v8[1];
      v11 = v12;
    }
  }
  return 0;
}

// Function: sub_31751A0
// Address: 0x31751a0
//
__int64 __fastcall sub_31751A0(__int64 a1, _BYTE *a2)
{
  _BYTE *v2; // r12
  __int64 result; // rax
  int v4; // eax
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  _BYTE *v9; // rdi
  int v10; // eax
  int v11; // r8d

  v2 = a2;
  if ( *a2 <= 0x15u )
    return (__int64)v2;
  result = sub_2A66C60(*(__int64 **)(a1 + 56), (__int64)a2);
  if ( !result )
  {
    v4 = *(_DWORD *)(a1 + 88);
    v5 = *(_QWORD *)(a1 + 72);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v8 = (_QWORD *)(v5 + 16LL * v7);
      v9 = (_BYTE *)*v8;
      if ( v2 == (_BYTE *)*v8 )
        return v8[1];
      v10 = 1;
      while ( v9 != (_BYTE *)-4096LL )
      {
        v11 = v10 + 1;
        v7 = v6 & (v10 + v7);
        v8 = (_QWORD *)(v5 + 16LL * v7);
        v9 = (_BYTE *)*v8;
        if ( v2 == (_BYTE *)*v8 )
          return v8[1];
        v10 = v11;
      }
    }
    return 0;
  }
  return result;
}

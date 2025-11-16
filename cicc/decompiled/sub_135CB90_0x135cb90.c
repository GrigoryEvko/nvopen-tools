// Function: sub_135CB90
// Address: 0x135cb90
//
__int64 __fastcall sub_135CB90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // ebx
  __m128i v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+10h] [rbp-30h]

  v10 = 0u;
  v11 = 0;
  sub_14A8180(a2, &v10, 0);
  v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v3 = *(_QWORD *)(a2 + 24 * (2 - v2));
  v4 = -1;
  if ( *(_BYTE *)(v3 + 16) == 13 )
  {
    v4 = *(_QWORD *)(v3 + 24);
    if ( *(_DWORD *)(v3 + 32) > 0x40u )
      v4 = *(_QWORD *)v4;
  }
  v5 = sub_135C460(a1, *(_QWORD *)(a2 - 24 * v2), v4, &v10, 2);
  result = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(result + 16) )
    BUG();
  if ( *(_DWORD *)(result + 36) == 137 )
  {
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    result = 3 * (3 - v7);
    v8 = *(_QWORD *)(a2 + 24 * (3 - v7));
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 > 0x40 )
    {
      result = sub_16A57B0(v8 + 24);
      if ( v9 == (_DWORD)result )
        return result;
      goto LABEL_8;
    }
    if ( *(_QWORD *)(v8 + 24) )
LABEL_8:
      *(_BYTE *)(v5 + 67) |= 0x80u;
  }
  return result;
}

// Function: sub_135CC90
// Address: 0x135cc90
//
__int64 __fastcall sub_135CC90(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // ebx
  __m128i v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+10h] [rbp-40h]

  v2 = -1;
  v12 = 0u;
  v13 = 0;
  sub_14A8180(a2, &v12, 0);
  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v4 = *(_QWORD *)(a2 + 24 * (2 - v3));
  if ( *(_BYTE *)(v4 + 16) == 13 )
  {
    v2 = *(_QWORD *)(v4 + 24);
    if ( *(_DWORD *)(v4 + 32) > 0x40u )
      v2 = *(_QWORD *)v2;
  }
  v5 = sub_135C460(a1, *(_QWORD *)(a2 + 24 * (1 - v3)), v2, &v12, 1);
  v6 = sub_135C460(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v2, &v12, 2);
  v7 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v7 + 16) )
    BUG();
  result = *(_DWORD *)(v7 + 36) & 0xFFFFFFFD;
  if ( (_DWORD)result == 133 )
  {
    v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    result = 3 * (3 - v9);
    v10 = *(_QWORD *)(a2 + 24 * (3 - v9));
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 > 0x40 )
    {
      result = sub_16A57B0(v10 + 24);
      if ( v11 == (_DWORD)result )
        return result;
      goto LABEL_8;
    }
    if ( *(_QWORD *)(v10 + 24) )
    {
LABEL_8:
      *(_BYTE *)(v5 + 67) |= 0x80u;
      *(_BYTE *)(v6 + 67) |= 0x80u;
    }
  }
  return result;
}

// Function: sub_31B7CD0
// Address: 0x31b7cd0
//
__int64 __fastcall sub_31B7CD0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rcx
  bool v6; // of
  __int64 v7; // rax
  int v8; // esi
  __int64 *v9; // rdi

  v3 = *(_QWORD *)(a2 + 160);
  v4 = *(_QWORD *)(a2 + 144);
  v5 = *(_QWORD *)(a2 + 120);
  if ( *(_DWORD *)(a2 + 168) == 1 )
  {
    v9 = (__int64 *)(v5 + 8);
    goto LABEL_5;
  }
  v6 = __OFSUB__(v4, v3);
  v7 = v4 - v3;
  v8 = *(_DWORD *)(a2 + 152);
  v9 = (__int64 *)(v5 + 8);
  if ( v6 )
  {
    if ( v3 <= 0 )
    {
      if ( !v8 )
        goto LABEL_5;
    }
    else if ( !v8 )
    {
      goto LABEL_9;
    }
    goto LABEL_8;
  }
  if ( v8 )
  {
LABEL_8:
    if ( v8 >= 0 )
      goto LABEL_5;
    goto LABEL_9;
  }
  if ( -(int)qword_5035A48 <= v7 )
  {
LABEL_5:
    sub_318DF30(v9);
    return 0;
  }
LABEL_9:
  LOBYTE(v2) = *(_DWORD *)(v5 + 16) != 0;
  sub_318DFC0((__int64)v9);
  return v2;
}

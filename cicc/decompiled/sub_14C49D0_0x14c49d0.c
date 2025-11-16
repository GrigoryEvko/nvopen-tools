// Function: sub_14C49D0
// Address: 0x14c49d0
//
__int64 __fastcall sub_14C49D0(_BYTE *a1)
{
  unsigned __int8 v2; // al
  __int64 v4; // rdi
  _DWORD *v5; // rcx
  _DWORD *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // r12d
  _BYTE *v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  _BYTE v12[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a1[16];
  if ( v2 > 0x10u )
  {
    if ( v2 != 85 )
      return 0;
    v4 = *((_QWORD *)a1 - 3);
    v11 = 0x1000000000LL;
    v10 = v12;
    sub_15FAA20(v4, &v10);
    v5 = &v10[4 * (unsigned int)v11];
    if ( v10 != (_BYTE *)v5 )
    {
      v6 = v10;
      while ( (unsigned int)(*v6 + 1) <= 1 )
      {
        if ( v5 == ++v6 )
          goto LABEL_12;
      }
      if ( v10 != v12 )
        _libc_free((unsigned __int64)v10);
      return 0;
    }
LABEL_12:
    if ( v10 != v12 )
      _libc_free((unsigned __int64)v10);
    v7 = *((_QWORD *)a1 - 9);
    if ( *(_BYTE *)(v7 + 16) != 84 )
      return 0;
    v8 = *(_QWORD *)(v7 - 24);
    if ( *(_BYTE *)(v8 + 16) != 13 )
      return 0;
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      if ( *(_QWORD *)(v8 + 24) )
        return 0;
    }
    else if ( v9 != (unsigned int)sub_16A57B0(v8 + 24) )
    {
      return 0;
    }
    return *(_QWORD *)(v7 - 48);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
      return 0;
    return sub_15A1020(a1);
  }
}

// Function: sub_140E6D0
// Address: 0x140e6d0
//
__int64 __fastcall sub_140E6D0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned __int8 v9; // al
  __int64 *v10; // rax
  char v11; // dl
  unsigned __int8 v12; // al
  __int64 *v14; // rsi
  unsigned int v15; // edi
  __int64 *v16; // rcx
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-28h]

  v6 = sub_15A9570(*(_QWORD *)a2, *a3);
  *(_DWORD *)(a2 + 20) = v6;
  v18 = v6;
  if ( v6 > 0x40 )
    sub_16A4EF0(&v17, 0, 0);
  else
    v17 = 0;
  if ( *(_DWORD *)(a2 + 32) > 0x40u )
  {
    v7 = *(_QWORD *)(a2 + 24);
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
  *(_QWORD *)(a2 + 24) = v17;
  *(_DWORD *)(a2 + 32) = v18;
  v8 = sub_1649C60(a3);
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 > 0x17u )
  {
    v10 = *(__int64 **)(a2 + 48);
    if ( *(__int64 **)(a2 + 56) != v10 )
    {
LABEL_8:
      sub_16CCBA0(a2 + 40, v8);
      if ( v11 )
        goto LABEL_9;
      goto LABEL_19;
    }
    v14 = &v10[*(unsigned int *)(a2 + 68)];
    v15 = *(_DWORD *)(a2 + 68);
    if ( v10 == v14 )
    {
LABEL_32:
      if ( v15 < *(_DWORD *)(a2 + 64) )
      {
        *(_DWORD *)(a2 + 68) = v15 + 1;
        *v14 = v8;
        ++*(_QWORD *)(a2 + 40);
LABEL_9:
        v12 = *(_BYTE *)(v8 + 16);
        if ( v12 <= 0x17u )
        {
          if ( v12 != 5 || *(_WORD *)(v8 + 18) != 32 )
            goto LABEL_11;
        }
        else if ( v12 != 56 )
        {
LABEL_11:
          sub_140E600(a1, a2, v8);
          return a1;
        }
        goto LABEL_23;
      }
      goto LABEL_8;
    }
    v16 = 0;
    while ( v8 != *v10 )
    {
      if ( *v10 == -2 )
        v16 = v10;
      if ( v14 == ++v10 )
      {
        if ( !v16 )
          goto LABEL_32;
        *v16 = v8;
        --*(_DWORD *)(a2 + 72);
        ++*(_QWORD *)(a2 + 40);
        goto LABEL_9;
      }
    }
LABEL_19:
    *(_DWORD *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    *(_DWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    return a1;
  }
  if ( v9 != 17 )
  {
    switch ( v9 )
    {
      case 0xFu:
        sub_140C910(a1, a2, v8);
        return a1;
      case 1u:
        sub_140EDF0(a1, a2, v8);
        return a1;
      case 3u:
        sub_140CA20(a1, a2, v8);
        return a1;
      case 9u:
        sub_140CF00(a1, a2);
        return a1;
    }
    if ( v9 == 5 && *(_WORD *)(v8 + 18) == 32 )
    {
LABEL_23:
      sub_140EC20(a1, a2, v8);
      return a1;
    }
    goto LABEL_19;
  }
  sub_140BFD0(a1, a2, v8);
  return a1;
}

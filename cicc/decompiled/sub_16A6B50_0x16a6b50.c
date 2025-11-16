// Function: sub_16A6B50
// Address: 0x16a6b50
//
__int64 __fastcall sub_16A6B50(__int64 a1, unsigned __int64 **a2, unsigned __int64 a3)
{
  unsigned int v4; // ebx
  unsigned int v5; // eax
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // [rsp+8h] [rbp-38h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v4 = *((_DWORD *)a2 + 2);
  v11 = a3;
  if ( v4 > 0x40 )
  {
    v5 = v4 - sub_16A57B0((__int64)a2);
    v6 = ((unsigned __int64)v5 + 63) >> 6;
    if ( !v6 )
      goto LABEL_9;
    v7 = v11;
    if ( v11 == 1 )
    {
      *(_DWORD *)(a1 + 8) = v4;
      sub_16A4FD0(a1, (const void **)a2);
      return a1;
    }
    if ( v5 > 0x40 )
    {
LABEL_7:
      if ( v6 != 1 )
      {
        v13 = v4;
        sub_16A4EF0((__int64)&v12, 0, 0);
        sub_16A6110((__int64 *)*a2, v6, (__int64 *)&v11, 1u, v12, 0);
        *(_DWORD *)(a1 + 8) = v13;
        *(_QWORD *)a1 = v12;
        return a1;
      }
      v10 = **a2;
      *(_DWORD *)(a1 + 8) = v4;
      sub_16A4EF0(a1, v10 / v7, 0);
      return a1;
    }
    if ( v11 <= **a2 )
    {
      if ( v11 != **a2 )
        goto LABEL_7;
      *(_DWORD *)(a1 + 8) = v4;
      sub_16A4EF0(a1, 1, 0);
    }
    else
    {
LABEL_9:
      *(_DWORD *)(a1 + 8) = v4;
      sub_16A4EF0(a1, 0, 0);
    }
    return a1;
  }
  v9 = (unsigned __int64)*a2;
  *(_DWORD *)(a1 + 8) = v4;
  *(_QWORD *)a1 = (v9 / v11) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
  return a1;
}

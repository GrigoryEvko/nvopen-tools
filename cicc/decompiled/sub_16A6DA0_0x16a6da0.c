// Function: sub_16A6DA0
// Address: 0x16a6da0
//
void __fastcall sub_16A6DA0(__int64 **a1, unsigned __int64 a2, unsigned __int64 *a3, unsigned __int64 *a4)
{
  unsigned int v6; // ebx
  unsigned int v7; // eax
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 *v10; // rtt
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rsi
  unsigned int v13; // ecx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]

  v6 = *((_DWORD *)a1 + 2);
  v16 = a2;
  if ( v6 > 0x40 )
  {
    v7 = v6 - sub_16A57B0((__int64)a1);
    v8 = ((unsigned __int64)v7 + 63) >> 6;
    if ( v8 )
    {
      if ( v16 == 1 )
      {
        sub_16A51C0((__int64)a3, (__int64)a1);
        *a4 = 0;
        return;
      }
      if ( v7 > 0x40 )
        goto LABEL_5;
      v9 = **a1;
      if ( v16 > v9 )
      {
        *a4 = v9;
        v18 = v6;
        sub_16A4EF0((__int64)&v17, 0, 0);
        if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
          j_j___libc_free_0_0(*a3);
        *a3 = v17;
        *((_DWORD *)a3 + 2) = v18;
        return;
      }
      if ( v16 != v9 )
      {
LABEL_5:
        sub_16A5130(a3, v6);
        if ( v8 == 1 )
        {
          v12 = v16;
          v13 = *((_DWORD *)a3 + 2);
          v14 = **a1;
          v15 = v14 / v16;
          if ( v13 <= 0x40 )
          {
            *a3 = v15 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
          }
          else
          {
            *(_QWORD *)*a3 = v15;
            memset(
              (void *)(*a3 + 8),
              0,
              8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
            v12 = v16;
          }
          *a4 = v14 % v12;
        }
        else
        {
          sub_16A6110(*a1, v8, (__int64 *)&v16, 1u, *a3, (__int64)a4);
          memset((void *)(*a3 + 8 * v8), 0, 8 * ((unsigned int)(((unsigned __int64)v6 + 63) >> 6) - (unsigned int)v8));
        }
        return;
      }
      v18 = v6;
      sub_16A4EF0((__int64)&v17, 1, 0);
      if ( *((_DWORD *)a3 + 2) > 0x40u )
        goto LABEL_14;
    }
    else
    {
      v18 = v6;
      sub_16A4EF0((__int64)&v17, 0, 0);
      if ( *((_DWORD *)a3 + 2) > 0x40u )
      {
LABEL_14:
        if ( *a3 )
          j_j___libc_free_0_0(*a3);
      }
    }
    *a3 = v17;
    *((_DWORD *)a3 + 2) = v18;
    *a4 = 0;
    return;
  }
  v10 = *a1;
  *a4 = (unsigned __int64)*a1 % a2;
  v11 = ((unsigned __int64)v10 / a2) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v6);
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    j_j___libc_free_0_0(*a3);
  *a3 = v11;
  *((_DWORD *)a3 + 2) = v6;
}

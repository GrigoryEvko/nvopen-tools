// Function: sub_C45A90
// Address: 0xc45a90
//
void __fastcall sub_C45A90(__int64 **a1, unsigned __int64 a2, unsigned __int64 *a3, unsigned __int64 *a4)
{
  unsigned int v6; // ebx
  unsigned int v7; // eax
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // [rsp+0h] [rbp-50h]
  unsigned __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-38h]

  v6 = *((_DWORD *)a1 + 2);
  v14 = a2;
  if ( v6 > 0x40 )
  {
    v7 = v6 - sub_C444A0((__int64)a1);
    v8 = ((unsigned __int64)v7 + 63) >> 6;
    if ( v8 )
    {
      if ( v14 == 1 )
      {
        sub_C43990((__int64)a3, (__int64)a1);
        *a4 = 0;
        return;
      }
      if ( v7 > 0x40 )
        goto LABEL_5;
      v9 = **a1;
      if ( v14 > v9 )
      {
        *a4 = v9;
        v16 = v6;
        sub_C43690((__int64)&v15, 0, 0);
        if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
          j_j___libc_free_0_0(*a3);
        *a3 = v15;
        *((_DWORD *)a3 + 2) = v16;
        return;
      }
      if ( v14 != v9 )
      {
LABEL_5:
        sub_C43900(a3, v6);
        if ( v8 == 1 )
        {
          v11 = **a1;
          v12 = v11 / v14;
          if ( *((_DWORD *)a3 + 2) <= 0x40u )
          {
            *a3 = v12;
            sub_C43640(a3);
          }
          else
          {
            *(_QWORD *)*a3 = v12;
            memset(
              (void *)(*a3 + 8),
              0,
              8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
          }
          *a4 = v11 % v14;
        }
        else
        {
          sub_C44DF0(*a1, v8, (__int64 *)&v14, 1u, *a3, (__int64)a4);
          memset((void *)(*a3 + 8 * v8), 0, 8 * ((unsigned int)(((unsigned __int64)v6 + 63) >> 6) - (unsigned int)v8));
        }
        return;
      }
      v16 = v6;
      sub_C43690((__int64)&v15, 1, 0);
      if ( *((_DWORD *)a3 + 2) > 0x40u )
        goto LABEL_14;
    }
    else
    {
      v16 = v6;
      sub_C43690((__int64)&v15, 0, 0);
      if ( *((_DWORD *)a3 + 2) > 0x40u )
      {
LABEL_14:
        if ( *a3 )
          j_j___libc_free_0_0(*a3);
      }
    }
    *a3 = v15;
    *((_DWORD *)a3 + 2) = v16;
    *a4 = 0;
    return;
  }
  v10 = (unsigned __int64)*a1 / a2;
  *a4 = (unsigned __int64)*a1 % a2;
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
  {
    v13 = v10;
    j_j___libc_free_0_0(*a3);
    v10 = v13;
  }
  *((_DWORD *)a3 + 2) = v6;
  *a3 = v10;
}

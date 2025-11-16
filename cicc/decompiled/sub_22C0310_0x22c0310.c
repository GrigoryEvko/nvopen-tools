// Function: sub_22C0310
// Address: 0x22c0310
//
void __fastcall sub_22C0310(__int64 a1, unsigned __int8 *a2, char a3)
{
  char v4; // di
  int v5; // ecx
  unsigned __int64 v7; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-48h]
  unsigned __int64 v9; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-38h]
  unsigned __int64 v11; // [rsp+20h] [rbp-30h]
  unsigned int v12; // [rsp+28h] [rbp-28h]

  v4 = *(_BYTE *)a1;
  v5 = *a2;
  if ( (unsigned int)(v5 - 12) > 1 )
  {
    if ( v4 != 2 )
    {
      if ( (_BYTE)v5 == 17 )
      {
        v8 = *((_DWORD *)a2 + 8);
        if ( v8 > 0x40 )
          sub_C43780((__int64)&v7, (const void **)a2 + 3);
        else
          v7 = *((_QWORD *)a2 + 3);
        sub_AADBC0((__int64)&v9, (__int64 *)&v7);
        sub_22C00F0(a1, (__int64)&v9, a3, 0, 1u);
        if ( v12 > 0x40 && v11 )
          j_j___libc_free_0_0(v11);
        if ( v10 > 0x40 && v9 )
          j_j___libc_free_0_0(v9);
        if ( v8 > 0x40 && v7 )
          j_j___libc_free_0_0(v7);
      }
      else
      {
        *(_BYTE *)a1 = 2;
        *(_QWORD *)(a1 + 8) = a2;
      }
    }
  }
  else if ( v4 != 1 )
  {
    *(_BYTE *)a1 = 1;
  }
}

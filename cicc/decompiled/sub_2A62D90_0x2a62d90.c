// Function: sub_2A62D90
// Address: 0x2a62d90
//
__int64 __fastcall sub_2A62D90(__int64 a1)
{
  const void **v1; // r12
  unsigned int v3; // r14d
  unsigned __int64 v4; // r13
  bool v5; // cc
  int v6; // eax
  const void *v7; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-38h]
  const void *v9; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-28h]

  LODWORD(v1) = 1;
  if ( *(_BYTE *)a1 != 2 )
  {
    LODWORD(v1) = 0;
    if ( (unsigned __int8)(*(_BYTE *)a1 - 4) <= 1u )
    {
      v10 = *(_DWORD *)(a1 + 16);
      v1 = &v9;
      if ( v10 > 0x40 )
        sub_C43780((__int64)&v9, (const void **)(a1 + 8));
      else
        v9 = *(const void **)(a1 + 8);
      sub_C46A40((__int64)&v9, 1);
      v3 = v10;
      v4 = (unsigned __int64)v9;
      v10 = 0;
      v5 = *(_DWORD *)(a1 + 32) <= 0x40u;
      v8 = v3;
      v7 = v9;
      if ( v5 )
      {
        LOBYTE(v1) = *(_QWORD *)(a1 + 24) == (_QWORD)v9;
      }
      else
      {
        LOBYTE(v6) = sub_C43C50(a1 + 24, &v7);
        LODWORD(v1) = v6;
      }
      if ( v3 > 0x40 )
      {
        if ( v4 )
        {
          j_j___libc_free_0_0(v4);
          if ( v10 > 0x40 )
          {
            if ( v9 )
              j_j___libc_free_0_0((unsigned __int64)v9);
          }
        }
      }
    }
  }
  return (unsigned int)v1;
}

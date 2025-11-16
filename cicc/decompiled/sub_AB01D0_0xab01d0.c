// Function: sub_AB01D0
// Address: 0xab01d0
//
char __fastcall sub_AB01D0(__int64 a1, __int64 a2)
{
  char v2; // r8
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // r14
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // r15
  char v10; // [rsp+Fh] [rbp-61h]
  char v11; // [rsp+Fh] [rbp-61h]
  char v12; // [rsp+Fh] [rbp-61h]
  __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-58h]
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-48h]
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-38h]
  __int64 v19; // [rsp+40h] [rbp-30h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-28h]

  v2 = sub_AAF760(a1);
  LOBYTE(v3) = 0;
  if ( !v2 )
  {
    LOBYTE(v3) = sub_AAF760(a2);
    if ( !(_BYTE)v3 )
    {
      v16 = *(_DWORD *)(a1 + 24);
      if ( v16 > 0x40 )
        sub_C43780(&v15, a1 + 16);
      else
        v15 = *(_QWORD *)(a1 + 16);
      sub_C46B40(&v15, a1);
      v4 = v16;
      v5 = v15;
      v16 = 0;
      v6 = *(_DWORD *)(a2 + 24);
      v14 = v4;
      v13 = v15;
      v20 = v6;
      if ( v6 > 0x40 )
        sub_C43780(&v19, a2 + 16);
      else
        v19 = *(_QWORD *)(a2 + 16);
      sub_C46B40(&v19, a2);
      v7 = v20;
      v8 = v19;
      v20 = 0;
      v18 = v7;
      v17 = v19;
      v3 = (unsigned int)sub_C49970(&v13, &v17) >> 31;
      if ( v7 > 0x40 )
      {
        if ( v8 )
        {
          v10 = v3;
          j_j___libc_free_0_0(v8);
          LOBYTE(v3) = v10;
          if ( v20 > 0x40 )
          {
            if ( v19 )
            {
              j_j___libc_free_0_0(v19);
              LOBYTE(v3) = v10;
            }
          }
        }
      }
      if ( v4 > 0x40 && v5 )
      {
        v11 = v3;
        j_j___libc_free_0_0(v5);
        LOBYTE(v3) = v11;
      }
      if ( v16 > 0x40 && v15 )
      {
        v12 = v3;
        j_j___libc_free_0_0(v15);
        LOBYTE(v3) = v12;
      }
    }
  }
  return v3;
}

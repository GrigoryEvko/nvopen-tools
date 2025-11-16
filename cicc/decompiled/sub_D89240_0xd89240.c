// Function: sub_D89240
// Address: 0xd89240
//
__int64 __fastcall sub_D89240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-58h]
  __int64 v19; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-48h]
  __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-38h]

  if ( sub_AAF7D0(a5) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 24), 0);
  }
  else
  {
    sub_D890C0((__int64)&v15, a2, a3, a4);
    if ( sub_AAF7D0((__int64)&v15) || sub_AAF760((__int64)&v15) || sub_AB01B0((__int64)&v15) )
    {
      v8 = *(_DWORD *)(a2 + 40);
      *(_DWORD *)(a1 + 8) = v8;
      if ( v8 > 0x40 )
        sub_C43780(a1, (const void **)(a2 + 32));
      else
        *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
      v9 = *(_DWORD *)(a2 + 56);
      *(_DWORD *)(a1 + 24) = v9;
      if ( v9 > 0x40 )
        sub_C43780(a1 + 16, (const void **)(a2 + 48));
      else
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 48);
    }
    else
    {
      sub_D87200((__int64)&v19, (__int64)&v15, a5);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      v15 = v19;
      v10 = v20;
      v20 = 0;
      v16 = v10;
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      v17 = v21;
      v11 = v22;
      v22 = 0;
      v18 = v11;
      sub_969240(&v21);
      sub_969240(&v19);
      if ( sub_AAF7D0((__int64)&v15) || sub_AAF760((__int64)&v15) || sub_AB01B0((__int64)&v15) )
      {
        sub_AAF450(a1, a2 + 32);
      }
      else
      {
        v12 = v16;
        v16 = 0;
        *(_DWORD *)(a1 + 8) = v12;
        *(_QWORD *)a1 = v15;
        v13 = v18;
        v18 = 0;
        *(_DWORD *)(a1 + 24) = v13;
        *(_QWORD *)(a1 + 16) = v17;
      }
    }
    sub_969240(&v17);
    sub_969240(&v15);
  }
  return a1;
}

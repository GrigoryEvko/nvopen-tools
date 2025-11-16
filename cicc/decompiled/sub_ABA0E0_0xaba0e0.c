// Function: sub_ABA0E0
// Address: 0xaba0e0
//
__int64 __fastcall sub_ABA0E0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-88h]
  __int64 v16; // [rsp+30h] [rbp-80h]
  unsigned int v17; // [rsp+38h] [rbp-78h]
  __int64 v18[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v19[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v20; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+68h] [rbp-48h]
  __int64 v22; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+78h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else if ( sub_AAF760(a2) && sub_AAF760(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
  }
  else
  {
    sub_AB4F10((__int64)&v14, a2, a3);
    if ( (a4 & 2) != 0 )
    {
      sub_AB9F50((__int64)v18, a2, a3);
      sub_AB2160((__int64)&v20, (__int64)&v14, (__int64)v18, a5);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
      v14 = v20;
      v10 = v21;
      v21 = 0;
      v15 = v10;
      if ( v17 > 0x40 && v16 )
        j_j___libc_free_0_0(v16);
      v16 = v22;
      v11 = v23;
      v23 = 0;
      v17 = v11;
      sub_969240(&v22);
      sub_969240(&v20);
      sub_969240(v19);
      sub_969240(v18);
    }
    if ( (a4 & 1) != 0 )
    {
      sub_AB9DC0((__int64)v18, a2, a3);
      sub_AB2160((__int64)&v20, (__int64)&v14, (__int64)v18, a5);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
      v14 = v20;
      v12 = v21;
      v21 = 0;
      v15 = v12;
      if ( v17 > 0x40 && v16 )
        j_j___libc_free_0_0(v16);
      v16 = v22;
      v13 = v23;
      v23 = 0;
      v17 = v13;
      sub_969240(&v22);
      sub_969240(&v20);
      sub_969240(v19);
      sub_969240(v18);
    }
    v9 = v15;
    v15 = 0;
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)a1 = v14;
    *(_DWORD *)(a1 + 24) = v17;
    *(_QWORD *)(a1 + 16) = v16;
    sub_969240(&v14);
  }
  return a1;
}

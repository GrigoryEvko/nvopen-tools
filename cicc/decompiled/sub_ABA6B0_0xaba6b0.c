// Function: sub_ABA6B0
// Address: 0xaba6b0
//
__int64 __fastcall sub_ABA6B0(__int64 a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  int v13; // ebx
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-88h]
  __int64 v18; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-78h]
  __int64 v20[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v21[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v22; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-48h]
  __int64 v24; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+78h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0(a3) )
  {
    if ( sub_AAF760(a2) && sub_AAF760(a3) )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
      return a1;
    }
    sub_AB51C0((__int64)&v16, a2, a3);
    if ( (a4 & 2) != 0 )
    {
      sub_ABA520((__int64)v20, a2, a3);
      sub_AB2160((__int64)&v22, (__int64)&v16, (__int64)v20, a5);
      if ( v17 > 0x40 && v16 )
        j_j___libc_free_0_0(v16);
      v16 = v22;
      v11 = v23;
      v23 = 0;
      v17 = v11;
      if ( v19 > 0x40 && v18 )
        j_j___libc_free_0_0(v18);
      v18 = v24;
      v12 = v25;
      v25 = 0;
      v19 = v12;
      sub_969240(&v24);
      sub_969240(&v22);
      sub_969240(v21);
      sub_969240(v20);
    }
    if ( (a4 & 1) != 0 )
    {
      sub_AB0910((__int64)v20, a2);
      sub_AB0A00((__int64)&v22, a3);
      v13 = sub_C49970(v20, &v22);
      sub_969240(&v22);
      sub_969240(v20);
      if ( v13 < 0 )
      {
        sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
        goto LABEL_10;
      }
      sub_ABA390((__int64)v20, a2, a3);
      sub_AB2160((__int64)&v22, (__int64)&v16, (__int64)v20, a5);
      if ( v17 > 0x40 && v16 )
        j_j___libc_free_0_0(v16);
      v16 = v22;
      v14 = v23;
      v23 = 0;
      v17 = v14;
      if ( v19 > 0x40 && v18 )
        j_j___libc_free_0_0(v18);
      v18 = v24;
      v15 = v25;
      v25 = 0;
      v19 = v15;
      sub_969240(&v24);
      sub_969240(&v22);
      sub_969240(v21);
      sub_969240(v20);
    }
    v9 = v17;
    v17 = 0;
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)a1 = v16;
    v10 = v19;
    v19 = 0;
    *(_DWORD *)(a1 + 24) = v10;
    *(_QWORD *)(a1 + 16) = v18;
LABEL_10:
    sub_969240(&v18);
    sub_969240(&v16);
    return a1;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}

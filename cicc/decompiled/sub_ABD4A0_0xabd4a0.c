// Function: sub_ABD4A0
// Address: 0xabd4a0
//
__int64 __fastcall sub_ABD4A0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r14d
  unsigned int v4; // eax
  unsigned int v5; // eax
  __int64 v6; // r10
  unsigned __int64 v7; // rdx
  __int64 v8; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-88h]
  __int64 v10; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-78h]
  __int64 v12; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-68h]
  __int64 v14; // [rsp+40h] [rbp-60h] BYREF
  __int64 v15; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+58h] [rbp-48h]
  __int64 v17[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    v3 = *(_DWORD *)(a2 + 8);
    v9 = v3;
    if ( v3 > 0x40 )
      sub_C43690(&v8, 0, 0);
    else
      v8 = 0;
    if ( sub_AAF760(a2) )
    {
      sub_9691E0((__int64)&v10, v3, v3, 0, 0);
      sub_C46A40(&v10, 1);
      v4 = v11;
      v11 = 0;
      v13 = v4;
      v12 = v10;
      v16 = v9;
      if ( v9 > 0x40 )
        sub_C43780(&v15, &v8);
      else
        v15 = v8;
      sub_9875E0(a1, &v15, &v12);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      if ( v13 > 0x40 && v12 )
        j_j___libc_free_0_0(v12);
      if ( v11 > 0x40 && v10 )
        j_j___libc_free_0_0(v10);
    }
    else if ( sub_AAFBB0(a2) )
    {
      sub_9691E0((__int64)&v15, v3, v3 + 1, 0, 0);
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
      {
        v6 = (unsigned int)sub_C44500(a2);
      }
      else if ( v5 )
      {
        v6 = 64;
        if ( *(_QWORD *)a2 << (64 - (unsigned __int8)v5) != -1 )
        {
          _BitScanReverse64(&v7, ~(*(_QWORD *)a2 << (64 - (unsigned __int8)v5)));
          v6 = (int)(v7 ^ 0x3F);
        }
      }
      else
      {
        v6 = 0;
      }
      sub_9691E0((__int64)&v10, v3, v6, 0, 0);
      sub_AADC30((__int64)&v12, (__int64)&v10, &v15);
      sub_969240(&v10);
      sub_969240(&v15);
      sub_AAEC90((__int64)&v15, &v8, a2 + 16);
      sub_AB3510(a1, (__int64)&v12, (__int64)&v15, 0);
      sub_969240(v17);
      sub_969240(&v15);
      sub_969240(&v14);
      sub_969240(&v12);
    }
    else
    {
      sub_AAEC90(a1, (_QWORD *)a2, a2 + 16);
    }
    if ( v9 > 0x40 && v8 )
      j_j___libc_free_0_0(v8);
  }
  return a1;
}

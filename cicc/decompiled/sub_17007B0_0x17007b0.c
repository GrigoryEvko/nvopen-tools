// Function: sub_17007B0
// Address: 0x17007b0
//
__int64 __fastcall sub_17007B0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 v6; // r12
  _BYTE **v8; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v9; // [rsp+20h] [rbp-D0h]
  _BYTE *v10; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v11; // [rsp+38h] [rbp-B8h]
  _BYTE v12[176]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 24LL);
  if ( v2 == sub_16FF760 )
  {
    v10 = v12;
    v11 = 0x8000000000LL;
    BUG();
  }
  v3 = v2();
  v10 = v12;
  v4 = *(_QWORD *)(v3 + 768);
  v11 = 0x8000000000LL;
  sub_1700740(a1, (__int64)&v10, a2, v4, 0);
  v5 = *(_QWORD *)(v3 + 760);
  v9 = 262;
  v8 = &v10;
  v6 = sub_38BF510(v5, &v8);
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
  return v6;
}

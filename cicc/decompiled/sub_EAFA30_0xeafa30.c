// Function: sub_EAFA30
// Address: 0xeafa30
//
__int64 __fastcall sub_EAFA30(__int64 a1)
{
  bool v1; // zf
  unsigned int v2; // r12d
  _QWORD *v4; // rax
  _BYTE v5[16]; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v6; // [rsp+10h] [rbp-50h]
  _QWORD *v7; // [rsp+20h] [rbp-40h] BYREF
  size_t v8; // [rsp+28h] [rbp-38h]
  _QWORD v9[6]; // [rsp+30h] [rbp-30h] BYREF

  v1 = *(_BYTE *)(a1 + 869) == 0;
  v7 = v9;
  v8 = 0;
  LOBYTE(v9[0]) = 0;
  if ( v1 && (unsigned __int8)sub_EA2540(a1) || (v2 = sub_EAE3B0((_QWORD *)a1, &v7), (_BYTE)v2) )
  {
    v2 = 1;
  }
  else
  {
    v4 = sub_E66210(*(_QWORD *)(a1 + 224), (__int64)&v7);
    sub_E5F7F0((__int64)v5, (__int64)v4, v7, v8);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 536LL))(*(_QWORD *)(a1 + 232), v6, 4);
  }
  if ( v7 != v9 )
    j_j___libc_free_0(v7, v9[0] + 1LL);
  return v2;
}

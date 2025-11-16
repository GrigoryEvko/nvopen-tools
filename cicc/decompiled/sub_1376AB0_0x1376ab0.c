// Function: sub_1376AB0
// Address: 0x1376ab0
//
__int64 __fastcall sub_1376AB0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v10; // [rsp+18h] [rbp-78h]
  __int64 v11; // [rsp+20h] [rbp-70h]
  void *v12; // [rsp+30h] [rbp-60h]
  _QWORD v13[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v14; // [rsp+48h] [rbp-48h]
  __int64 v15; // [rsp+50h] [rbp-40h]

  *(_QWORD *)a1 = &unk_49E8AA8;
  v2 = *(_QWORD *)(a1 + 416);
  if ( v2 != *(_QWORD *)(a1 + 408) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 248);
  if ( v3 != *(_QWORD *)(a1 + 240) )
    _libc_free(v3);
  j___libc_free_0(*(_QWORD *)(a1 + 200));
  v4 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v4 )
  {
    v6 = *(_QWORD **)(a1 + 168);
    v9[0] = 2;
    v9[1] = 0;
    v10 = -8;
    v7 = &v6[5 * v4];
    v11 = 0;
    v13[0] = 2;
    v13[1] = 0;
    v14 = -16;
    v12 = &unk_49E8A80;
    v15 = 0;
    do
    {
      v8 = v6[3];
      *v6 = &unk_49EE2B0;
      if ( v8 != 0 && v8 != -8 && v8 != -16 )
        sub_1649B30(v6 + 1);
      v6 += 5;
    }
    while ( v7 != v6 );
    v12 = &unk_49EE2B0;
    if ( v14 != -8 && v14 != 0 && v14 != -16 )
      sub_1649B30(v13);
    if ( v10 != -8 && v10 != 0 && v10 != -16 )
      sub_1649B30(v9);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 168));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0(a1);
}

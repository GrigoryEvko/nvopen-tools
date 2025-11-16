// Function: sub_1952890
// Address: 0x1952890
//
void *__fastcall sub_1952890(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  __int64 v11; // rax
  _QWORD v12[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v13; // [rsp+28h] [rbp-78h]
  __int64 v14; // [rsp+30h] [rbp-70h]
  void *v15; // [rsp+40h] [rbp-60h]
  _QWORD v16[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v17; // [rsp+58h] [rbp-48h]
  __int64 v18; // [rsp+60h] [rbp-40h]

  *(_QWORD *)a1 = off_49F3AC0;
  j___libc_free_0(*(_QWORD *)(a1 + 392));
  v2 = *(_QWORD *)(a1 + 232);
  if ( v2 != *(_QWORD *)(a1 + 224) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 200);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 256);
    if ( v4 != *(_QWORD *)(v3 + 248) )
      _libc_free(v4);
    v5 = *(_QWORD *)(v3 + 88);
    if ( v5 != *(_QWORD *)(v3 + 80) )
      _libc_free(v5);
    j___libc_free_0(*(_QWORD *)(v3 + 40));
    if ( *(_DWORD *)(v3 + 24) )
    {
      v13 = -8;
      v14 = 0;
      v17 = -16;
      v18 = 0;
      v15 = &unk_49E8A80;
      v8 = *(unsigned int *)(v3 + 24);
      v12[0] = 2;
      v12[1] = 0;
      v16[0] = 2;
      v16[1] = 0;
      v9 = *(_QWORD **)(v3 + 8);
      v10 = &v9[5 * v8];
      if ( v9 != v10 )
      {
        do
        {
          v11 = v9[3];
          *v9 = &unk_49EE2B0;
          if ( v11 != 0 && v11 != -8 && v11 != -16 )
            sub_1649B30(v9 + 1);
          v9 += 5;
        }
        while ( v10 != v9 );
        v15 = &unk_49EE2B0;
        if ( v17 != 0 && v17 != -16 && v17 != -8 )
          sub_1649B30(v16);
      }
      if ( v13 != -8 && v13 != 0 && v13 != -16 )
        sub_1649B30(v12);
    }
    j___libc_free_0(*(_QWORD *)(v3 + 8));
    j_j___libc_free_0(v3, 408);
  }
  v6 = *(_QWORD *)(a1 + 192);
  if ( v6 )
  {
    sub_1368A00(*(__int64 **)(a1 + 192));
    j_j___libc_free_0(v6, 8);
  }
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}

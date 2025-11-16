// Function: sub_13E6970
// Address: 0x13e6970
//
void __fastcall sub_13E6970(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r14
  __int64 v7; // rax
  _QWORD v8[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v9; // [rsp+18h] [rbp-78h]
  __int64 v10; // [rsp+20h] [rbp-70h]
  void *v11; // [rsp+30h] [rbp-60h]
  _QWORD v12[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v13; // [rsp+48h] [rbp-48h]
  __int64 v14; // [rsp+50h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 256);
    if ( v2 != *(_QWORD *)(v1 + 248) )
      _libc_free(v2);
    v3 = *(_QWORD *)(v1 + 88);
    if ( v3 != *(_QWORD *)(v1 + 80) )
      _libc_free(v3);
    j___libc_free_0(*(_QWORD *)(v1 + 40));
    if ( *(_DWORD *)(v1 + 24) )
    {
      v9 = -8;
      v10 = 0;
      v13 = -16;
      v14 = 0;
      v11 = &unk_49E8A80;
      v4 = *(unsigned int *)(v1 + 24);
      v8[0] = 2;
      v8[1] = 0;
      v12[0] = 2;
      v12[1] = 0;
      v5 = *(_QWORD **)(v1 + 8);
      v6 = &v5[5 * v4];
      if ( v5 != v6 )
      {
        do
        {
          v7 = v5[3];
          *v5 = &unk_49EE2B0;
          if ( v7 != 0 && v7 != -8 && v7 != -16 )
            sub_1649B30(v5 + 1);
          v5 += 5;
        }
        while ( v6 != v5 );
        v11 = &unk_49EE2B0;
        if ( v13 != -8 && v13 != 0 && v13 != -16 )
          sub_1649B30(v12);
      }
      if ( v9 != 0 && v9 != -8 && v9 != -16 )
        sub_1649B30(v8);
    }
    j___libc_free_0(*(_QWORD *)(v1 + 8));
    j_j___libc_free_0(v1, 440);
  }
}

// Function: sub_13E6B30
// Address: 0x13e6b30
//
__int64 __fastcall sub_13E6B30(_QWORD *a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r15
  __int64 v9; // rax
  _QWORD v10[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v11; // [rsp+28h] [rbp-78h]
  __int64 v12; // [rsp+30h] [rbp-70h]
  void *v13; // [rsp+40h] [rbp-60h]
  _QWORD v14[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v15; // [rsp+58h] [rbp-48h]
  __int64 v16; // [rsp+60h] [rbp-40h]

  v2 = a1[20];
  *a1 = &unk_49EA848;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 256);
    if ( v3 != *(_QWORD *)(v2 + 248) )
      _libc_free(v3);
    v4 = *(_QWORD *)(v2 + 88);
    if ( v4 != *(_QWORD *)(v2 + 80) )
      _libc_free(v4);
    j___libc_free_0(*(_QWORD *)(v2 + 40));
    if ( *(_DWORD *)(v2 + 24) )
    {
      v11 = -8;
      v12 = 0;
      v15 = -16;
      v16 = 0;
      v13 = &unk_49E8A80;
      v6 = *(unsigned int *)(v2 + 24);
      v10[0] = 2;
      v10[1] = 0;
      v14[0] = 2;
      v14[1] = 0;
      v7 = *(_QWORD **)(v2 + 8);
      v8 = &v7[5 * v6];
      if ( v7 != v8 )
      {
        do
        {
          v9 = v7[3];
          *v7 = &unk_49EE2B0;
          if ( v9 != 0 && v9 != -8 && v9 != -16 )
            sub_1649B30(v7 + 1);
          v7 += 5;
        }
        while ( v8 != v7 );
        v13 = &unk_49EE2B0;
        if ( v15 != -8 && v15 != 0 && v15 != -16 )
          sub_1649B30(v14);
      }
      if ( v11 != 0 && v11 != -8 && v11 != -16 )
        sub_1649B30(v10);
    }
    j___libc_free_0(*(_QWORD *)(v2 + 8));
    j_j___libc_free_0(v2, 440);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}

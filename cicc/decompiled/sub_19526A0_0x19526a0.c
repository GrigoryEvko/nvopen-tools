// Function: sub_19526A0
// Address: 0x19526a0
//
void __fastcall sub_19526A0(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r14
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v10; // [rsp+18h] [rbp-78h]
  __int64 v11; // [rsp+20h] [rbp-70h]
  void *v12; // [rsp+30h] [rbp-60h]
  _QWORD v13[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v14; // [rsp+48h] [rbp-48h]
  __int64 v15; // [rsp+50h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 192);
  *(_QWORD *)(a1 + 192) = 0;
  if ( v1 )
  {
    sub_1368A00(v1);
    j_j___libc_free_0(v1, 8);
  }
  v2 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 200) = 0;
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
      v10 = -8;
      v11 = 0;
      v14 = -16;
      v15 = 0;
      v12 = &unk_49E8A80;
      v5 = *(unsigned int *)(v2 + 24);
      v9[0] = 2;
      v9[1] = 0;
      v13[0] = 2;
      v13[1] = 0;
      v6 = *(_QWORD **)(v2 + 8);
      v7 = &v6[5 * v5];
      if ( v6 != v7 )
      {
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
        if ( v14 != 0 && v14 != -16 && v14 != -8 )
          sub_1649B30(v13);
      }
      if ( v10 != -8 && v10 != 0 && v10 != -16 )
        sub_1649B30(v9);
    }
    j___libc_free_0(*(_QWORD *)(v2 + 8));
    j_j___libc_free_0(v2, 408);
  }
}

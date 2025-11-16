// Function: sub_14CA8A0
// Address: 0x14ca8a0
//
__int64 __fastcall sub_14CA8A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  __int64 v6; // r15
  __int64 i; // rax
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-98h]
  _QWORD v13[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h]
  void *v16; // [rsp+40h] [rbp-60h]
  _QWORD v17[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v18; // [rsp+58h] [rbp-48h]
  __int64 v19; // [rsp+60h] [rbp-40h]

  v1 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v1 )
  {
    v6 = *(_QWORD *)(a1 + 160);
    v13[0] = 2;
    v13[1] = 0;
    v14 = -8;
    v16 = &unk_49ECBD0;
    v15 = 0;
    v17[0] = 2;
    v17[1] = 0;
    v18 = -16;
    v19 = 0;
    v12 = v6 + 88 * v1;
    for ( i = -8; ; i = v14 )
    {
      v8 = *(_QWORD *)(v6 + 24);
      if ( v8 != i )
      {
        i = v18;
        if ( v8 != v18 )
        {
          v9 = *(_QWORD *)(v6 + 40);
          v10 = v9 + 32LL * *(unsigned int *)(v6 + 48);
          if ( v9 != v10 )
          {
            do
            {
              v11 = *(_QWORD *)(v10 - 16);
              v10 -= 32LL;
              if ( v11 != 0 && v11 != -8 && v11 != -16 )
                sub_1649B30(v10);
            }
            while ( v9 != v10 );
            v10 = *(_QWORD *)(v6 + 40);
          }
          if ( v10 != v6 + 56 )
            _libc_free(v10);
          i = *(_QWORD *)(v6 + 24);
        }
      }
      *(_QWORD *)v6 = &unk_49EE2B0;
      if ( i != -8 && i != 0 && i != -16 )
        sub_1649B30(v6 + 8);
      v6 += 88;
      if ( v12 == v6 )
        break;
    }
    v16 = &unk_49EE2B0;
    if ( v18 != -8 && v18 != 0 && v18 != -16 )
      sub_1649B30(v17);
    if ( v14 != -8 && v14 != 0 && v14 != -16 )
      sub_1649B30(v13);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 160));
  v2 = *(_QWORD *)(a1 + 8);
  v3 = v2 + 32LL * *(unsigned int *)(a1 + 16);
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 - 16);
      v3 -= 32LL;
      if ( v4 != -8 && v4 != 0 && v4 != -16 )
        sub_1649B30(v3);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 8);
  }
  if ( v3 != a1 + 24 )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 192);
}

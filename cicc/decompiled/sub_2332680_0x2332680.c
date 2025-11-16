// Function: sub_2332680
// Address: 0x2332680
//
void __fastcall sub_2332680(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 i; // rax
  __int64 v7; // rcx
  _QWORD *v8; // r14
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-98h]
  _QWORD v12[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v13; // [rsp+28h] [rbp-78h]
  __int64 v14; // [rsp+30h] [rbp-70h]
  void *v15; // [rsp+40h] [rbp-60h]
  _QWORD v16[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v17; // [rsp+58h] [rbp-48h]
  __int64 v18; // [rsp+60h] [rbp-40h]

  *(_QWORD *)a1 = &unk_4A0B1A0;
  v1 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v1 )
  {
    v5 = *(_QWORD *)(a1 + 176);
    v12[0] = 2;
    v12[1] = 0;
    v13 = -4096;
    v15 = &unk_49DDAE8;
    v14 = 0;
    v16[0] = 2;
    v16[1] = 0;
    v17 = -8192;
    v18 = 0;
    v11 = v5 + 88 * v1;
    for ( i = -4096; ; i = v13 )
    {
      v7 = *(_QWORD *)(v5 + 24);
      if ( v7 != i )
      {
        i = v17;
        if ( v7 != v17 )
        {
          v8 = *(_QWORD **)(v5 + 40);
          v9 = &v8[4 * *(unsigned int *)(v5 + 48)];
          if ( v8 != v9 )
          {
            do
            {
              v10 = *(v9 - 2);
              v9 -= 4;
              if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
                sub_BD60C0(v9);
            }
            while ( v8 != v9 );
            v9 = *(_QWORD **)(v5 + 40);
          }
          if ( v9 != (_QWORD *)(v5 + 56) )
            _libc_free((unsigned __int64)v9);
          i = *(_QWORD *)(v5 + 24);
        }
      }
      *(_QWORD *)v5 = &unk_49DB368;
      if ( i != -4096 && i != 0 && i != -8192 )
        sub_BD60C0((_QWORD *)(v5 + 8));
      v5 += 88;
      if ( v11 == v5 )
        break;
    }
    v15 = &unk_49DB368;
    sub_D68D70(v16);
    sub_D68D70(v12);
    v1 = *(unsigned int *)(a1 + 192);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 88 * v1, 8);
  v2 = *(_QWORD **)(a1 + 24);
  v3 = &v2[4 * *(unsigned int *)(a1 + 32)];
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(v3 - 2);
      v3 -= 4;
      if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
        sub_BD60C0(v3);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)(a1 + 24);
  }
  if ( v3 != (_QWORD *)(a1 + 40) )
    _libc_free((unsigned __int64)v3);
}

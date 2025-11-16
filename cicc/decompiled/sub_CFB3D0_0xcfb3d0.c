// Function: sub_CFB3D0
// Address: 0xcfb3d0
//
__int64 __fastcall sub_CFB3D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rax
  __int64 v8; // r15
  __int64 i; // rax
  __int64 v10; // rcx
  _QWORD *v11; // r14
  _QWORD *v12; // r12
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-98h]
  _QWORD v15[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v16; // [rsp+28h] [rbp-78h]
  __int64 v17; // [rsp+30h] [rbp-70h]
  void *v18; // [rsp+40h] [rbp-60h]
  _QWORD v19[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v2 )
  {
    v8 = *(_QWORD *)(a1 + 168);
    v15[0] = 2;
    v15[1] = 0;
    v16 = -4096;
    v18 = &unk_49DDAE8;
    v17 = 0;
    v19[0] = 2;
    v19[1] = 0;
    v20 = -8192;
    v21 = 0;
    v14 = v8 + 88 * v2;
    for ( i = -4096; ; i = v16 )
    {
      v10 = *(_QWORD *)(v8 + 24);
      if ( v10 != i )
      {
        i = v20;
        if ( v10 != v20 )
        {
          v11 = *(_QWORD **)(v8 + 40);
          v12 = &v11[4 * *(unsigned int *)(v8 + 48)];
          if ( v11 != v12 )
          {
            do
            {
              v13 = *(v12 - 2);
              v12 -= 4;
              LOBYTE(a2) = v13 != -4096;
              if ( ((v13 != 0) & (unsigned __int8)a2) != 0 && v13 != -8192 )
                sub_BD60C0(v12);
            }
            while ( v11 != v12 );
            v12 = *(_QWORD **)(v8 + 40);
          }
          if ( v12 != (_QWORD *)(v8 + 56) )
            _libc_free(v12, a2);
          i = *(_QWORD *)(v8 + 24);
        }
      }
      *(_QWORD *)v8 = &unk_49DB368;
      LOBYTE(a2) = i != 0;
      if ( i != -4096 && i != 0 && i != -8192 )
        sub_BD60C0((_QWORD *)(v8 + 8));
      v8 += 88;
      if ( v14 == v8 )
        break;
    }
    v18 = &unk_49DB368;
    if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
      sub_BD60C0(v19);
    if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
      sub_BD60C0(v15);
    v2 = *(unsigned int *)(a1 + 184);
  }
  v3 = 88 * v2;
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 88 * v2, 8);
  v4 = *(_QWORD **)(a1 + 16);
  v5 = &v4[4 * *(unsigned int *)(a1 + 24)];
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(v5 - 2);
      v5 -= 4;
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD60C0(v5);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 16);
  }
  if ( v5 != (_QWORD *)(a1 + 32) )
    _libc_free(v5, v3);
  return j_j___libc_free_0(a1, 200);
}

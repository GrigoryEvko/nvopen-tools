// Function: sub_24E4610
// Address: 0x24e4610
//
void __fastcall sub_24E4610(__int64 a1)
{
  bool v1; // zf
  __int64 v2; // rsi
  __int64 v3; // r12
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rsi
  _QWORD v14[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v15; // [rsp+18h] [rbp-78h]
  __int64 v16; // [rsp+20h] [rbp-70h]
  void *v17; // [rsp+30h] [rbp-60h]
  _QWORD v18[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v19; // [rsp+48h] [rbp-48h]
  __int64 v20; // [rsp+50h] [rbp-40h]

  v1 = *(_BYTE *)(a1 + 264) == 0;
  *(_QWORD *)a1 = &unk_4A16A60;
  if ( !v1 )
  {
    v10 = *(unsigned int *)(a1 + 256);
    *(_BYTE *)(a1 + 264) = 0;
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD **)(a1 + 240);
      v12 = &v11[2 * v10];
      do
      {
        if ( *v11 != -8192 && *v11 != -4096 )
        {
          v13 = v11[1];
          if ( v13 )
            sub_B91220((__int64)(v11 + 1), v13);
        }
        v11 += 2;
      }
      while ( v12 != v11 );
      v10 = *(unsigned int *)(a1 + 256);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 240), 16 * v10, 8);
  }
  v2 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v2 )
  {
    v5 = *(_QWORD **)(a1 + 208);
    v14[0] = 2;
    v14[1] = 0;
    v15 = -4096;
    v6 = &v5[8 * v2];
    v17 = &unk_49DD7B0;
    v7 = -4096;
    v16 = 0;
    v18[0] = 2;
    v18[1] = 0;
    v19 = -8192;
    v20 = 0;
    while ( 1 )
    {
      v8 = v5[3];
      if ( v8 != v7 )
      {
        v7 = v19;
        if ( v8 != v19 )
        {
          v9 = v5[7];
          if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          {
            sub_BD60C0(v5 + 5);
            v8 = v5[3];
          }
          v7 = v8;
        }
      }
      *v5 = &unk_49DB368;
      if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
        sub_BD60C0(v5 + 1);
      v5 += 8;
      if ( v6 == v5 )
        break;
      v7 = v15;
    }
    v17 = &unk_49DB368;
    if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
      sub_BD60C0(v18);
    if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
      sub_BD60C0(v14);
    v2 = *(unsigned int *)(a1 + 224);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 208), v2 << 6, 8);
  nullsub_61();
  v3 = a1 + 56;
  *(_QWORD *)(a1 + 168) = &unk_49DA100;
  nullsub_63();
  v4 = *(_QWORD *)(a1 + 40);
  if ( v4 != v3 )
    _libc_free(v4);
}

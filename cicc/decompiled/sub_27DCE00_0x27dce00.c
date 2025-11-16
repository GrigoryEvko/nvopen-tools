// Function: sub_27DCE00
// Address: 0x27dce00
//
__int64 __fastcall sub_27DCE00(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  __int64 v11; // rsi
  _QWORD v12[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  __int64 v14; // [rsp+20h] [rbp-70h]
  void *v15; // [rsp+30h] [rbp-60h]
  _QWORD v16[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v17; // [rsp+48h] [rbp-48h]
  __int64 v18; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 64) )
  {
    v8 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 64) = 0;
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD **)(a1 + 40);
      v10 = &v9[2 * v8];
      do
      {
        if ( *v9 != -8192 && *v9 != -4096 )
        {
          v11 = v9[1];
          if ( v11 )
            sub_B91220((__int64)(v9 + 1), v11);
        }
        v9 += 2;
      }
      while ( v10 != v9 );
      v8 = *(unsigned int *)(a1 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 16 * v8, 8);
  }
  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v12[0] = 2;
    v12[1] = 0;
    v13 = -4096;
    v4 = &v3[8 * v1];
    v15 = &unk_49DD7B0;
    v5 = -4096;
    v14 = 0;
    v16[0] = 2;
    v16[1] = 0;
    v17 = -8192;
    v18 = 0;
    while ( 1 )
    {
      v6 = v3[3];
      if ( v6 != v5 )
      {
        v5 = v17;
        if ( v6 != v17 )
        {
          v7 = v3[7];
          if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
          {
            sub_BD60C0(v3 + 5);
            v6 = v3[3];
          }
          v5 = v6;
        }
      }
      *v3 = &unk_49DB368;
      if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
        sub_BD60C0(v3 + 1);
      v3 += 8;
      if ( v4 == v3 )
        break;
      v5 = v13;
    }
    v15 = &unk_49DB368;
    if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
      sub_BD60C0(v16);
    if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
      sub_BD60C0(v12);
    v1 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), v1 << 6, 8);
}

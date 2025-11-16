// Function: sub_1B46DA0
// Address: 0x1b46da0
//
__int64 __fastcall sub_1B46DA0(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
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
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD **)(a1 + 40);
      v10 = &v9[2 * v8];
      do
      {
        if ( *v9 != -8 && *v9 != -4 )
        {
          v11 = v9[1];
          if ( v11 )
            sub_161E7C0((__int64)(v9 + 1), v11);
        }
        v9 += 2;
      }
      while ( v10 != v9 );
    }
    j___libc_free_0(*(_QWORD *)(a1 + 40));
  }
  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v12[0] = 2;
    v12[1] = 0;
    v13 = -8;
    v4 = &v3[8 * v1];
    v15 = &unk_49E6B50;
    v5 = -8;
    v14 = 0;
    v16[0] = 2;
    v16[1] = 0;
    v17 = -16;
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
          if ( v7 != 0 && v7 != -8 && v7 != -16 )
          {
            sub_1649B30(v3 + 5);
            v6 = v3[3];
          }
          v5 = v6;
        }
      }
      *v3 = &unk_49EE2B0;
      if ( v5 != -8 && v5 != 0 && v5 != -16 )
        sub_1649B30(v3 + 1);
      v3 += 8;
      if ( v4 == v3 )
        break;
      v5 = v13;
    }
    v15 = &unk_49EE2B0;
    if ( v17 != -8 && v17 != 0 && v17 != -16 )
      sub_1649B30(v16);
    if ( v13 != 0 && v13 != -8 && v13 != -16 )
      sub_1649B30(v12);
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}

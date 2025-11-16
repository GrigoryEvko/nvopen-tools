// Function: sub_2C16D20
// Address: 0x2c16d20
//
__int64 __fastcall sub_2C16D20(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r9
  __int64 v5; // r15
  __int64 v7; // [rsp+0h] [rbp-70h]
  int v8; // [rsp+Ch] [rbp-64h]
  __int64 v9; // [rsp+18h] [rbp-58h] BYREF
  __int64 v10; // [rsp+20h] [rbp-50h] BYREF
  __int64 v11; // [rsp+28h] [rbp-48h] BYREF
  __int64 v12; // [rsp+30h] [rbp-40h] BYREF
  __int64 v13; // [rsp+38h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 48);
  v2 = *v1;
  v3 = v1[1];
  v8 = *(_DWORD *)(a1 + 156);
  v9 = *(_QWORD *)(a1 + 88);
  if ( v9 )
    sub_2AAAFA0(&v9);
  v5 = sub_22077B0(0xA8u);
  if ( v5 )
  {
    v7 = *(_QWORD *)(a1 + 160);
    v10 = v9;
    if ( v9 )
    {
      sub_2AAAFA0(&v10);
      v12 = v2;
      v13 = v3;
      v11 = v10;
      if ( v10 )
        sub_2AAAFA0(&v11);
    }
    else
    {
      v12 = v2;
      v13 = v3;
      v11 = 0;
    }
    sub_2AAF4A0(v5, 13, &v12, 2, &v11, v4);
    sub_9C6650(&v11);
    *(_BYTE *)(v5 + 152) = 4;
    *(_QWORD *)v5 = &unk_4A23258;
    *(_QWORD *)(v5 + 96) = &unk_4A232C8;
    *(_QWORD *)(v5 + 40) = &unk_4A23290;
    *(_DWORD *)(v5 + 156) = v8;
    sub_9C6650(&v10);
    *(_QWORD *)v5 = &unk_4A24260;
    *(_QWORD *)(v5 + 96) = &unk_4A242E0;
    *(_QWORD *)(v5 + 40) = &unk_4A242A8;
    *(_QWORD *)(v5 + 160) = v7;
  }
  sub_9C6650(&v9);
  return v5;
}

// Function: sub_2D64B90
// Address: 0x2d64b90
//
void __fastcall sub_2D64B90(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD v7[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v8; // [rsp+18h] [rbp-68h]
  __int64 v9; // [rsp+20h] [rbp-60h]
  void *v10; // [rsp+30h] [rbp-50h]
  _QWORD v11[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v12; // [rsp+48h] [rbp-38h]
  __int64 v13; // [rsp+50h] [rbp-30h]

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v7[0] = 2;
    v7[1] = 0;
    v8 = -4096;
    v3 = &v2[8 * v1];
    v10 = &unk_4A26638;
    v4 = -4096;
    v9 = 0;
    v11[0] = 2;
    v11[1] = 0;
    v12 = -8192;
    v13 = 0;
    while ( 1 )
    {
      v5 = v2[3];
      if ( v5 != v4 )
      {
        v4 = v12;
        if ( v5 != v12 )
        {
          v6 = v2[7];
          if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          {
            sub_BD60C0(v2 + 5);
            v5 = v2[3];
          }
          v4 = v5;
        }
      }
      *v2 = &unk_49DB368;
      if ( v4 != -4096 && v4 != 0 && v4 != -8192 )
        sub_BD60C0(v2 + 1);
      v2 += 8;
      if ( v3 == v2 )
        break;
      v4 = v8;
    }
    v10 = &unk_49DB368;
    if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
      sub_BD60C0(v11);
    if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
      sub_BD60C0(v7);
  }
}

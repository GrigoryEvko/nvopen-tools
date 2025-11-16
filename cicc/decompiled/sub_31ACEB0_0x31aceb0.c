// Function: sub_31ACEB0
// Address: 0x31aceb0
//
bool __fastcall sub_31ACEB0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r13
  char v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v10; // [rsp+8h] [rbp-88h]
  _QWORD v11[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v12; // [rsp+20h] [rbp-70h]
  int v13; // [rsp+28h] [rbp-68h]
  __int64 v14; // [rsp+30h] [rbp-60h]
  __int64 v15; // [rsp+38h] [rbp-58h]
  _BYTE *v16; // [rsp+40h] [rbp-50h]
  __int64 v17; // [rsp+48h] [rbp-48h]
  _BYTE v18[64]; // [rsp+50h] [rbp-40h] BYREF

  v1 = sub_AA5930(**(_QWORD **)(*a1 + 32));
  v10 = v2;
  v3 = v1;
  while ( v10 != v3 )
  {
    v6 = a1[2];
    v7 = *a1;
    v11[0] = 6;
    v11[1] = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    v16 = v18;
    v17 = 0x200000000LL;
    v4 = sub_1023CA0(v3, v7, v6, (__int64)v11, 0);
    if ( v4 )
    {
      if ( v13 == 1 )
        sub_31AC420((__int64)a1, v3, (__int64)v11, (__int64)(a1 + 43), v8);
      else
        v4 = 0;
    }
    if ( v16 != v18 )
      _libc_free((unsigned __int64)v16);
    if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
      sub_BD60C0(v11);
    if ( !v4 )
      break;
    if ( !v3 )
      BUG();
    v5 = *(_QWORD *)(v3 + 32);
    if ( !v5 )
      BUG();
    v3 = 0;
    if ( *(_BYTE *)(v5 - 24) == 84 )
      v3 = v5 - 24;
  }
  return v10 == v3;
}

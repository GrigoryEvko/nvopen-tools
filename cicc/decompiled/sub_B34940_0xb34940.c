// Function: sub_B34940
// Address: 0xb34940
//
__int64 __fastcall sub_B34940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-58h]
  int v8; // [rsp+Ch] [rbp-54h]
  _QWORD v9[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v10[32]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v11; // [rsp+40h] [rbp-20h]

  if ( !a3 )
  {
    v5 = sub_BCB2E0(*(_QWORD *)(a1 + 72));
    a3 = sub_ACD640(v5, -1, 0);
  }
  v9[0] = a3;
  v11 = 257;
  v3 = *(_QWORD *)(a2 + 8);
  v8 = 0;
  v6 = v3;
  v9[1] = a2;
  return sub_B33D10(a1, 0xD3u, (__int64)&v6, 1, (int)v9, 2, v7, (__int64)v10);
}

// Function: sub_2A7D090
// Address: 0x2a7d090
//
__int64 __fastcall sub_2A7D090(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edx
  _QWORD v12[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13; // [rsp+10h] [rbp-30h]
  _QWORD v14[2]; // [rsp+18h] [rbp-28h] BYREF
  __int64 v15; // [rsp+28h] [rbp-18h]

  result = *(unsigned int *)(a1 + 24);
  v5 = *(unsigned __int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v12[0] = 0;
  v12[1] = 0;
  v13 = -4096;
  v6 = &v5[10 * result];
  v14[0] = 0;
  v14[1] = 0;
  v15 = -4096;
  if ( v6 != v5 )
  {
    do
    {
      if ( v5 )
      {
        *v5 = 0;
        v5[1] = 0;
        v7 = v13;
        v8 = v13 == -4096;
        v5[2] = v13;
        if ( v7 != 0 && !v8 && v7 != -8192 )
          sub_BD6050(v5, v12[0] & 0xFFFFFFFFFFFFFFF8LL);
        v5[3] = 0;
        v5[4] = 0;
        v9 = v15;
        v8 = v15 == 0;
        v5[5] = v15;
        if ( v9 != -4096 && !v8 && v9 != -8192 )
          sub_BD6050(v5 + 3, v14[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v5 += 10;
    }
    while ( v5 != v6 );
    LODWORD(v10) = v15;
    if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
      v10 = sub_BD60C0(v14);
    v11 = v13;
    LOBYTE(v10) = v13 != -4096;
    LOBYTE(a4) = v13 != 0;
    LOBYTE(v11) = v13 != -8192;
    result = v11 & a4 & (unsigned int)v10;
    if ( (_BYTE)result )
      return sub_BD60C0(v12);
  }
  return result;
}

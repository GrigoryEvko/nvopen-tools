// Function: sub_272DAD0
// Address: 0x272dad0
//
__int64 __fastcall sub_272DAD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rsi
  int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v3 = a2 - a1;
  v4 = a1;
  v5 = 0xCF3CF3CF3CF3CF3DLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(a3 + 144);
    v7 = *(_QWORD *)(v6 + 8);
    v8 = v6 + 24;
    do
    {
      while ( 1 )
      {
        v9 = v5 >> 1;
        v10 = v4 + 168 * (v5 >> 1);
        v11 = *(_QWORD *)(v10 + 144);
        if ( v7 != *(_QWORD *)(v11 + 8) )
          break;
        v14 = v4;
        v15 = v8;
        v13 = sub_C49970(v8, (unsigned __int64 *)(v11 + 24));
        v8 = v15;
        v4 = v14;
        if ( v13 >= 0 )
        {
          v4 = v10 + 168;
          v5 = v5 - v9 - 1;
          goto LABEL_7;
        }
LABEL_3:
        v5 >>= 1;
        if ( v9 <= 0 )
          return v4;
      }
      if ( *(_DWORD *)(v6 + 32) < *(_DWORD *)(v11 + 32) )
        goto LABEL_3;
      v4 = v10 + 168;
      v5 = v5 - v9 - 1;
LABEL_7:
      ;
    }
    while ( v5 > 0 );
  }
  return v4;
}

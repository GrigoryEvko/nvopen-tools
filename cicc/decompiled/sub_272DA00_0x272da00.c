// Function: sub_272DA00
// Address: 0x272da00
//
__int64 __fastcall sub_272DA00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rdi
  int v12; // eax
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v14; // [rsp+8h] [rbp-38h]

  v3 = a2 - a1;
  v4 = a1;
  v5 = 0xCF3CF3CF3CF3CF3DLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(a3 + 144);
    v7 = *(_QWORD *)(v6 + 8);
    v8 = (unsigned __int64 *)(v6 + 24);
    do
    {
      while ( 1 )
      {
        v9 = v4 + 168 * (v5 >> 1);
        v10 = *(_QWORD *)(v9 + 144);
        if ( *(_QWORD *)(v10 + 8) != v7 )
          break;
        v13 = v4;
        v14 = v8;
        v12 = sub_C49970(v10 + 24, v8);
        v8 = v14;
        v4 = v13;
        if ( v12 >= 0 )
        {
          v5 >>= 1;
          goto LABEL_7;
        }
LABEL_3:
        v4 = v9 + 168;
        v5 = v5 - (v5 >> 1) - 1;
        if ( v5 <= 0 )
          return v4;
      }
      if ( *(_DWORD *)(v10 + 32) < *(_DWORD *)(v6 + 32) )
        goto LABEL_3;
      v5 >>= 1;
LABEL_7:
      ;
    }
    while ( v5 > 0 );
  }
  return v4;
}

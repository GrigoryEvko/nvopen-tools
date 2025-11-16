// Function: sub_22EB3F0
// Address: 0x22eb3f0
//
__int64 __fastcall sub_22EB3F0(__int64 a1, int a2, int a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 j; // r13
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 i; // r12
  int v14; // eax
  __int64 v15; // rsi
  int v16; // r14d
  __int64 v17; // rax
  int v18; // [rsp+4h] [rbp-8Ch]
  int v19; // [rsp+8h] [rbp-88h]
  __int64 v20; // [rsp+20h] [rbp-70h]
  __int64 v25; // [rsp+48h] [rbp-48h]
  __int64 v26; // [rsp+50h] [rbp-40h] BYREF
  __int64 v27[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a1 + 152);
  v19 = a1;
  v7 = *(_QWORD *)(v6 + 80);
  v20 = v6 + 72;
  if ( v6 + 72 == v7 )
  {
    j = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v7 + 32);
      if ( j != v7 + 24 )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 + 72 == v7 )
        break;
      if ( !v7 )
        BUG();
    }
  }
  result = a6;
  v18 = a6;
LABEL_8:
  while ( v20 != v7 )
  {
    if ( !j )
      BUG();
    v10 = *(_QWORD *)(j + 40);
    if ( v10 )
    {
      v11 = sub_B14240(v10);
      v25 = v12;
      for ( i = v11; i != v25; i = *(_QWORD *)(i + 8) )
      {
        if ( !*(_BYTE *)(i + 32) )
        {
          v14 = sub_B12000(i + 72);
          v15 = *(_QWORD *)(i + 24);
          v16 = v14;
          v26 = v15;
          if ( v15 )
          {
            sub_B96E90((__int64)&v26, v15, 1);
            v27[0] = v26;
            if ( v26 )
              sub_B96E90((__int64)v27, v26, 1);
          }
          else
          {
            v27[0] = 0;
          }
          sub_3142D90(v19, v16, (unsigned int)v27, a2, a3, v18, a4, a5);
          if ( v27[0] )
            sub_B91220((__int64)v27, v27[0]);
          if ( v26 )
            sub_B91220((__int64)&v26, v26);
        }
      }
    }
    for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v7 + 32) )
    {
      v17 = v7 - 24;
      if ( !v7 )
        v17 = 0;
      result = v17 + 48;
      if ( j != result )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( v20 == v7 )
        goto LABEL_8;
      if ( !v7 )
        BUG();
    }
  }
  return result;
}

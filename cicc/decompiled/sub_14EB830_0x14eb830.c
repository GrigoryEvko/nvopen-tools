// Function: sub_14EB830
// Address: 0x14eb830
//
unsigned __int64 *__fastcall sub_14EB830(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 *v4; // r13
  unsigned __int64 *v5; // r14
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rsi
  unsigned int v8; // edx
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 *i; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // r9
  unsigned __int64 *v16; // r11
  unsigned int v17; // r9d
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  __int64 v20; // rdi
  char v21; // cl
  unsigned __int64 v22; // rsi
  unsigned int v23; // r9d
  _QWORD v24[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(unsigned __int64 **)(a2 + 1528);
  v5 = *(unsigned __int64 **)(a2 + 1520);
  if ( v5 == v4 )
  {
LABEL_7:
    v10 = sub_16328F0(*(_QWORD *)(a2 + 440), "Linker Options", 14);
    if ( v10 )
    {
      v11 = sub_1632440(*(_QWORD *)(a2 + 440), "llvm.linker.options", 19);
      for ( i = (__int64 *)(v10 - 8LL * *(unsigned int *)(v10 + 8)); (__int64 *)v10 != i; ++i )
      {
        v13 = *i;
        sub_1623CA0(v11, v13);
      }
    }
    v14 = *(_QWORD *)(a2 + 1520);
    if ( v14 != *(_QWORD *)(a2 + 1528) )
      *(_QWORD *)(a2 + 1528) = v14;
    *a1 = 1;
  }
  else
  {
    while ( 1 )
    {
      v6 = *v5;
      *(_DWORD *)(a2 + 64) = 0;
      v7 = (v6 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a2 + 48) = v7;
      v8 = v6 & 0x3F;
      if ( (v6 & 0x3F) != 0 )
      {
        v15 = *(_QWORD *)(a2 + 40);
        if ( v7 >= v15 )
          goto LABEL_19;
        v16 = (unsigned __int64 *)(v7 + *(_QWORD *)(a2 + 32));
        if ( v15 >= v7 + 8 )
        {
          v19 = *v16;
          *(_QWORD *)(a2 + 48) = v7 + 8;
          v23 = 64;
        }
        else
        {
          *(_QWORD *)(a2 + 56) = 0;
          v17 = v15 - v7;
          if ( !v17 )
            goto LABEL_19;
          v18 = 0;
          v19 = 0;
          do
          {
            v20 = *((unsigned __int8 *)v16 + v18);
            v21 = 8 * v18++;
            v19 |= v20 << v21;
            *(_QWORD *)(a2 + 56) = v19;
          }
          while ( v17 != v18 );
          v22 = v17 + v7;
          v23 = 8 * v17;
          *(_QWORD *)(a2 + 48) = v22;
          *(_DWORD *)(a2 + 64) = v23;
          if ( v8 > v23 )
LABEL_19:
            sub_16BD130("Unexpected end of file", 1);
        }
        *(_DWORD *)(a2 + 64) = v23 - v8;
        *(_QWORD *)(a2 + 56) = v19 >> v8;
      }
      sub_1522BE0(v24, a2 + 608, 1);
      if ( (v24[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v4 == ++v5 )
        goto LABEL_7;
    }
    *a1 = v24[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  return a1;
}

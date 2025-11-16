// Function: sub_9CE050
// Address: 0x9ce050
//
unsigned __int64 *__fastcall sub_9CE050(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v5; // r13
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int8 v15; // al
  __int64 *v16; // r13
  __int64 v17; // rdx
  __int64 *v18; // r15
  __int64 v19; // rsi
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  char v22; // [rsp+28h] [rbp-38h]

  v5 = *(unsigned __int64 **)(a2 + 1680);
  v6 = *(unsigned __int64 **)(a2 + 1672);
  if ( v6 == v5 )
  {
LABEL_10:
    if ( !sub_BA8DC0(*(_QWORD *)(a2 + 440), "llvm.linker.options", 19) )
    {
      v13 = sub_BA91D0(*(_QWORD *)(a2 + 440), "Linker Options", 14);
      if ( v13 )
      {
        v14 = sub_BA8E40(*(_QWORD *)(a2 + 440), "llvm.linker.options", 19);
        v15 = *(_BYTE *)(v13 - 16);
        if ( (v15 & 2) != 0 )
        {
          v16 = *(__int64 **)(v13 - 32);
          v17 = *(unsigned int *)(v13 - 24);
        }
        else
        {
          v17 = (*(_WORD *)(v13 - 16) >> 6) & 0xF;
          v16 = (__int64 *)(v13 - 8LL * ((v15 >> 2) & 0xF) - 16);
        }
        v18 = &v16[v17];
        while ( v18 != v16 )
        {
          v19 = *v16++;
          sub_B979A0(v14, v19);
        }
      }
    }
    v12 = *(_QWORD *)(a2 + 1672);
    if ( v12 != *(_QWORD *)(a2 + 1680) )
      *(_QWORD *)(a2 + 1680) = v12;
    *a1 = 1;
  }
  else
  {
    while ( 1 )
    {
      v7 = *v6;
      *(_DWORD *)(a2 + 64) = 0;
      *(_QWORD *)(a2 + 48) = (v7 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
      v8 = v7 & 0x3F;
      if ( (_DWORD)v8 )
      {
        sub_9C66D0((__int64)&v21, a2 + 32, v8, a4);
        if ( (v22 & 1) != 0 )
        {
          v22 &= ~2u;
          v9 = v21;
          v21 = 0;
          v20 = v9 | 1;
        }
        else
        {
          v20 = 1;
        }
        v10 = v20 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v20 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
      }
      sub_A14940(&v21, a2 + 808, 1);
      v10 = v21 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v21 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      if ( v5 == ++v6 )
        goto LABEL_10;
    }
    *a1 = v10 | 1;
  }
  return a1;
}

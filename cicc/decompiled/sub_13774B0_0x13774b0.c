// Function: sub_13774B0
// Address: 0x13774b0
//
__int64 __fastcall sub_13774B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r13d
  unsigned int v8; // r15d
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // r10d
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned int i; // eax
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // [rsp+8h] [rbp-58h]
  char v24; // [rsp+1Bh] [rbp-45h]
  unsigned int v25; // [rsp+1Ch] [rbp-44h]
  _DWORD v26[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = sub_157EBA0(a2);
  if ( !v5 || (v6 = v5, (v7 = sub_15F4D60(v5)) == 0) )
  {
    v20 = sub_157EBA0(a2);
    v18 = 0;
    if ( v20 )
      v18 = (unsigned int)sub_15F4D60(v20);
LABEL_18:
    sub_16AF710(v26, 1, v18);
    return v26[0];
  }
  v25 = 0;
  v8 = 0;
  v24 = 0;
  v22 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
  do
  {
    if ( a3 == sub_15F4DF0(v6, v8) )
    {
      v9 = *(unsigned int *)(a1 + 56);
      if ( (_DWORD)v9 )
      {
        v10 = *(_QWORD *)(a1 + 40);
        v11 = 1;
        v12 = ((((37 * v8) | v22) - 1 - ((unsigned __int64)(37 * v8) << 32)) >> 22)
            ^ (((37 * v8) | v22) - 1 - ((unsigned __int64)(37 * v8) << 32));
        v13 = ((9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13)))) >> 15)
            ^ (9 * (((v12 - 1 - (v12 << 13)) >> 8) ^ (v12 - 1 - (v12 << 13))));
        for ( i = (v9 - 1) & (((v13 - 1 - (v13 << 27)) >> 31) ^ (v13 - 1 - ((_DWORD)v13 << 27))); ; i = (v9 - 1) & v16 )
        {
          v15 = v10 + 24LL * i;
          if ( a2 == *(_QWORD *)v15 && *(_DWORD *)(v15 + 8) == v8 )
            break;
          if ( *(_QWORD *)v15 == -8 && *(_DWORD *)(v15 + 8) == -1 )
            goto LABEL_4;
          v16 = v11 + i;
          ++v11;
        }
        if ( v15 != v10 + 24 * v9 )
        {
          v21 = *(unsigned int *)(v15 + 16);
          if ( v21 + (unsigned __int64)v25 > 0x80000000 )
          {
            v25 = 0x80000000;
            v24 = 1;
          }
          else
          {
            v24 = 1;
            v25 += v21;
          }
        }
      }
    }
LABEL_4:
    ++v8;
  }
  while ( v7 != v8 );
  v17 = sub_157EBA0(a2);
  v18 = 0;
  if ( v17 )
    v18 = (unsigned int)sub_15F4D60(v17);
  if ( !v24 )
    goto LABEL_18;
  v26[0] = v25;
  return v26[0];
}

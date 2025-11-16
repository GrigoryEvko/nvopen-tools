// Function: sub_1377370
// Address: 0x1377370
//
__int64 __fastcall sub_1377370(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rbp
  __int64 v7; // rdx
  __int64 v8; // rsi
  int v9; // r11d
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // r8
  unsigned int i; // eax
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v19; // [rsp-8h] [rbp-8h]

  v7 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 40);
    v9 = 1;
    v10 = ((((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(unsigned int)(37 * a3) << 32)) >> 22)
        ^ (((unsigned int)(37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * a3) << 32));
    v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
        ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
    for ( i = (v7 - 1) & (((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27))); ; i = (v7 - 1) & v14 )
    {
      v13 = v8 + 24LL * i;
      if ( a2 == *(_QWORD *)v13 && a3 == *(_DWORD *)(v13 + 8) )
        break;
      if ( *(_QWORD *)v13 == -8 && *(_DWORD *)(v13 + 8) == -1 )
        goto LABEL_10;
      v14 = v9 + i;
      ++v9;
    }
    if ( v13 != v8 + 24 * v7 )
      return *(unsigned int *)(v13 + 16);
  }
LABEL_10:
  v19 = v3;
  v16 = sub_157EBA0(a2);
  v17 = 0;
  if ( v16 )
    v17 = (unsigned int)sub_15F4D60(v16);
  sub_16AF710(&v18, 1, v17);
  return v18;
}

// Function: sub_19170B0
// Address: 0x19170b0
//
__int64 __fastcall sub_19170B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r11d
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rsi
  unsigned int i; // eax
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned int v16; // r15d
  char v18; // r8
  __int64 v19; // rax
  int v20; // edi
  unsigned int v21; // esi
  int v22; // edx
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  unsigned int v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h]
  unsigned int v26; // [rsp+20h] [rbp-40h]

  v8 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 160);
    v10 = 1;
    v11 = (((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4) | ((unsigned __int64)(37 * a4) << 32))
          - 1
          - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)) >> 22)
        ^ ((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4) | ((unsigned __int64)(37 * a4) << 32))
         - 1
         - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32));
    v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
        ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
    for ( i = (v8 - 1) & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = (v8 - 1) & v15 )
    {
      v14 = v9 + 24LL * i;
      if ( a4 == *(_DWORD *)v14 && a2 == *(_QWORD *)(v14 + 8) )
        break;
      if ( *(_DWORD *)v14 == -1 && *(_QWORD *)(v14 + 8) == -8 )
        goto LABEL_11;
      v15 = v10 + i;
      ++v10;
    }
    if ( v14 != v9 + 24 * v8 )
      return *(unsigned int *)(v14 + 16);
  }
LABEL_11:
  v24 = a4;
  v25 = a2;
  v26 = sub_19172C0(a1, a2, a3, a4);
  v16 = v26;
  v18 = sub_190ED80(a1 + 152, (int *)&v24, &v23);
  v19 = v23;
  if ( v18 )
    return v16;
  v20 = *(_DWORD *)(a1 + 168);
  v21 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 152);
  v22 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v21 )
  {
    v21 *= 2;
  }
  else if ( v21 - *(_DWORD *)(a1 + 172) - v22 > v21 >> 3 )
  {
    goto LABEL_14;
  }
  sub_1916E10(a1 + 152, v21);
  sub_190ED80(a1 + 152, (int *)&v24, &v23);
  v19 = v23;
  v22 = *(_DWORD *)(a1 + 168) + 1;
LABEL_14:
  *(_DWORD *)(a1 + 168) = v22;
  if ( *(_DWORD *)v19 != -1 || *(_QWORD *)(v19 + 8) != -8 )
    --*(_DWORD *)(a1 + 172);
  *(_DWORD *)v19 = v24;
  *(_QWORD *)(v19 + 8) = v25;
  *(_DWORD *)(v19 + 16) = v26;
  return v16;
}

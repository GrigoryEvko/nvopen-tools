// Function: sub_1954CE0
// Address: 0x1954ce0
//
__int64 __fastcall sub_1954CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned int v7; // esi
  __int64 v8; // rbx
  __int64 *v9; // rdi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned int i; // eax
  __int64 *v13; // rdx
  __int64 v14; // r10
  unsigned int v15; // eax
  int v17; // eax
  int v18; // r10d
  __int64 v19; // rdx
  int v20; // [rsp+18h] [rbp-58h]
  __int64 *v21; // [rsp+28h] [rbp-48h] BYREF
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  __int64 v23; // [rsp+38h] [rbp-38h]

  v3 = a1 + 224;
  v22 = a2;
  v7 = *(_DWORD *)(a1 + 248);
  v23 = a3;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 224);
    goto LABEL_24;
  }
  v8 = *(_QWORD *)(a1 + 232);
  v20 = 1;
  v9 = 0;
  v10 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  for ( i = (v7 - 1) & (((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27))); ; i = (v7 - 1) & v15 )
  {
    v13 = (__int64 *)(v8 + 16LL * i);
    v14 = *v13;
    if ( a2 == *v13 && a3 == v13[1] )
      return 0;
    if ( v14 == -8 )
      break;
    if ( v14 == -16 && v13[1] == -16 && !v9 )
      v9 = (__int64 *)(v8 + 16LL * i);
LABEL_9:
    v15 = v20 + i;
    ++v20;
  }
  if ( v13[1] != -8 )
    goto LABEL_9;
  v17 = *(_DWORD *)(a1 + 240);
  if ( !v9 )
    v9 = v13;
  ++*(_QWORD *)(a1 + 224);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) < 3 * v7 )
  {
    v19 = a2;
    if ( v7 - *(_DWORD *)(a1 + 244) - v18 <= v7 >> 3 )
    {
      sub_1954A40(v3, v7);
      sub_1954760(v3, &v22, &v21);
      v9 = v21;
      v19 = v22;
      v18 = *(_DWORD *)(a1 + 240) + 1;
    }
    goto LABEL_18;
  }
LABEL_24:
  sub_1954A40(v3, 2 * v7);
  sub_1954760(v3, &v22, &v21);
  v9 = v21;
  v19 = v22;
  v18 = *(_DWORD *)(a1 + 240) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 240) = v18;
  if ( *v9 != -8 || v9[1] != -8 )
    --*(_DWORD *)(a1 + 244);
  *v9 = v19;
  v9[1] = v23;
  return sub_1954F50(a1, a2, a3);
}

// Function: sub_1E49600
// Address: 0x1e49600
//
void __fastcall sub_1E49600(__int64 *a1, __int64 a2, char a3, unsigned int a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  int v12; // r9d
  __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rdi
  _DWORD *v20; // rax
  int v25; // [rsp+30h] [rbp-50h]
  __int64 v26; // [rsp+38h] [rbp-48h]
  int v27; // [rsp+44h] [rbp-3Ch] BYREF
  _QWORD v28[7]; // [rsp+48h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a2 + 40);
  if ( (_DWORD)v7 )
  {
    v26 = a7 + 32LL * a4;
    v10 = 40 * v7;
    v11 = 0;
    while ( 1 )
    {
      v13 = v11 + *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)v13 )
        goto LABEL_5;
      v14 = *(unsigned int *)(v13 + 8);
      if ( (int)v14 >= 0 )
        goto LABEL_5;
      v27 = *(_DWORD *)(v13 + 8);
      v15 = a1[5];
      if ( (*(_BYTE *)(v13 + 3) & 0x10) != 0 )
      {
        v25 = sub_1E6B9A0(
                v15,
                *(_QWORD *)(*(_QWORD *)(v15 + 24) + 16 * (v14 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                byte_3F871B3,
                0);
        sub_1E310D0(v13, v25);
        sub_1E49390(v26, &v27)[1] = v25;
        if ( a3 )
          sub_1E42770(v27, v25, a1[115], a1[5], a1[266], v12);
        goto LABEL_5;
      }
      v16 = sub_1E69D00(v15, v14);
      v17 = sub_1E45EB0((__int64)a1, v16);
      v18 = sub_1E404B0(a6, v17);
      if ( v18 == -1 || a5 <= v18 )
        v19 = v26;
      else
        v19 = a7 + 32LL * (a4 - a5 + v18);
      if ( !(unsigned __int8)sub_1932870(v19, &v27, v28) )
      {
LABEL_5:
        v11 += 40;
        if ( v11 == v10 )
          return;
      }
      else
      {
        v11 += 40;
        v20 = sub_1E49390(v19, &v27);
        sub_1E310D0(v13, v20[1]);
        if ( v11 == v10 )
          return;
      }
    }
  }
}

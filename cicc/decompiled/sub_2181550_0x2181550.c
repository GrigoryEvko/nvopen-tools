// Function: sub_2181550
// Address: 0x2181550
//
__int64 __fastcall sub_2181550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  unsigned int v8; // eax
  _QWORD *v9; // rdx
  unsigned int v10; // r11d
  unsigned int v12; // esi
  int v13; // eax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // [rsp+8h] [rbp-68h]
  unsigned __int8 v30; // [rsp+18h] [rbp-58h]
  unsigned __int8 v31; // [rsp+23h] [rbp-4Dh]
  __int64 v33; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v34; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v35[7]; // [rsp+38h] [rbp-38h] BYREF

  v33 = a2;
  v8 = sub_1E1F3B0(a5, &v33, v35);
  v9 = (_QWORD *)v35[0];
  v10 = v8;
  if ( (_BYTE)v8 )
    return v10;
  v12 = *(_DWORD *)(a5 + 24);
  v13 = *(_DWORD *)(a5 + 16);
  ++*(_QWORD *)a5;
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v12 )
  {
    v30 = v10;
    v12 *= 2;
  }
  else
  {
    if ( v12 - *(_DWORD *)(a5 + 20) - v14 > v12 >> 3 )
      goto LABEL_5;
    v30 = v10;
  }
  sub_1E22DE0(a5, v12);
  sub_1E1F3B0(a5, &v33, v35);
  v9 = (_QWORD *)v35[0];
  v10 = v30;
  v14 = *(_DWORD *)(a5 + 16) + 1;
LABEL_5:
  *(_DWORD *)(a5 + 16) = v14;
  if ( *v9 != -8 )
    --*(_DWORD *)(a5 + 20);
  *v9 = v33;
  if ( a6 <= 0x32 )
  {
    v15 = v33;
    v16 = *(unsigned int *)(v33 + 40);
    if ( (_DWORD)v16 )
    {
      v29 = a5;
      v17 = 40 * v16;
      v31 = v10;
      v18 = a4;
      v19 = 0;
      while ( 1 )
      {
        v20 = v19 + *(_QWORD *)(v15 + 32);
        if ( !*(_BYTE *)v20 && (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
        {
          v21 = *(unsigned int *)(v18 + 24);
          v34 = *(_DWORD *)(v20 + 8);
          v22 = *(_QWORD *)(v18 + 8) + 4 * v21;
          v25 = sub_1DF91F0(v18, (int *)&v34, v35);
          v27 = v35[0];
          if ( !(_BYTE)v25 )
          {
            v23 = *(unsigned int *)(v18 + 24);
            v27 = *(_QWORD *)(v18 + 8) + 4 * v23;
          }
          if ( v22 == v27 )
          {
            v28 = sub_217E810(a1, v34, v23, v24, v25, v26);
            if ( !v28 || !(unsigned __int8)sub_2181550(a1, v28, v34, v18, v29, a6 + 1) )
              break;
          }
        }
        v19 += 40;
        if ( v19 == v17 )
          return 1;
        v15 = v33;
      }
      return v31;
    }
    else
    {
      return 1;
    }
  }
  return v10;
}

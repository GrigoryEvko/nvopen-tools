// Function: sub_262DB60
// Address: 0x262db60
//
_QWORD *__fastcall sub_262DB60(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // r9
  int v10; // r11d
  unsigned int v11; // eax
  __int64 *v12; // rdi
  __int64 *v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rdi
  int v17; // eax
  int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v21; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v22[4]; // [rsp+10h] [rbp-20h] BYREF

  v5 = a3;
  v22[0] = a2;
  v22[1] = a3;
  if ( a3 && *a2 == 1 )
  {
    v5 = a3 - 1;
    ++a2;
  }
  v6 = sub_B2F650((__int64)a2, v5);
  v7 = *(_DWORD *)(a1 + 24);
  v20 = v6;
  v8 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    v21 = 0;
LABEL_22:
    v7 *= 2;
    goto LABEL_23;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = 1;
  v11 = (v7 - 1) & (((0xBF58476D1CE4E5B9LL * v6) >> 31) ^ (484763065 * v6));
  v12 = 0;
  v13 = (__int64 *)(v9
                  + 56LL * ((v7 - 1) & ((unsigned int)((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (484763065 * (_DWORD)v8))));
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_6:
    v15 = v13 + 1;
    return sub_9D35B0(v15, (__int64)v22);
  }
  while ( v14 != -1 )
  {
    if ( v14 == -2 && !v12 )
      v12 = v13;
    v11 = (v7 - 1) & (v10 + v11);
    v13 = (__int64 *)(v9 + 56LL * v11);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_6;
    ++v10;
  }
  v17 = *(_DWORD *)(a1 + 16);
  if ( !v12 )
    v12 = v13;
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  v21 = v12;
  if ( 4 * (v17 + 1) >= 3 * v7 )
    goto LABEL_22;
  if ( v7 - *(_DWORD *)(a1 + 20) - v18 <= v7 >> 3 )
  {
LABEL_23:
    sub_9EB160(a1, v7);
    sub_262B770(a1, &v20, &v21);
    v8 = v20;
    v12 = v21;
    v18 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v12 != -1 )
    --*(_DWORD *)(a1 + 20);
  v19 = v12 + 2;
  *v12 = v8;
  v15 = v12 + 1;
  *((_DWORD *)v15 + 2) = 0;
  v15[2] = 0;
  v15[3] = v19;
  v15[4] = v19;
  v15[5] = 0;
  return sub_9D35B0(v15, (__int64)v22);
}

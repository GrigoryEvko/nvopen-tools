// Function: sub_1E62AD0
// Address: 0x1e62ad0
//
__int64 __fastcall sub_1E62AD0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r14
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v14; // r15
  __int64 v15; // rbx
  int v16; // eax
  int v17; // edx
  int v18; // r9d

  v3 = a1[3];
  sub_1E06620(v3);
  v4 = *(_QWORD *)(v3 + 1312);
  v5 = *(unsigned int *)(v4 + 48);
  if ( !(_DWORD)v5 )
    return 0;
  v6 = *(_QWORD *)(v4 + 32);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v17 = 1;
    while ( v9 != -8 )
    {
      v18 = v17 + 1;
      v7 = (v5 - 1) & (v17 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v17 = v18;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v6 + 16 * v5) || !v8[1] )
    return 0;
  v10 = a1[4];
  if ( !v10 )
    return 1;
  v11 = a1[3];
  v12 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  sub_1E06620(v11);
  if ( !sub_1E05550(*(_QWORD *)(v11 + 1312), v12, a2) )
    return 0;
  v14 = a1[3];
  sub_1E06620(v14);
  if ( !sub_1E05550(*(_QWORD *)(v14 + 1312), v10, a2) )
    return 1;
  v15 = a1[3];
  sub_1E06620(v15);
  LOBYTE(v16) = sub_1E05550(*(_QWORD *)(v15 + 1312), v12, v10);
  return v16 ^ 1u;
}

// Function: sub_2281E00
// Address: 0x2281e00
//
_QWORD *__fastcall sub_2281E00(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rcx
  int v4; // eax
  int v5; // edi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r8
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r10
  _QWORD *v16; // r12
  __int64 v17; // rdi
  _QWORD *result; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  int v23; // edx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  int v29; // r11d
  int v30; // r9d
  __int64 v31[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(*a1 + 104LL);
  v4 = *(_DWORD *)(*a1 + 120LL);
  if ( !v4 )
  {
LABEL_18:
    v9 = 0;
    goto LABEL_4;
  }
  v5 = v4 - 1;
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v3 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v28 = 1;
    while ( v8 != -4096 )
    {
      v30 = v28 + 1;
      v6 = v5 & (v28 + v6);
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v28 = v30;
    }
    goto LABEL_18;
  }
LABEL_3:
  v9 = v7[1];
LABEL_4:
  v10 = a1[1];
  v31[0] = v9;
  v11 = *(unsigned int *)(v10 + 96);
  v12 = *(_QWORD *)(v10 + 80);
  if ( !(_DWORD)v11 )
    goto LABEL_13;
  v13 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( v9 != *v14 )
  {
    v23 = 1;
    while ( v15 != -4096 )
    {
      v29 = v23 + 1;
      v13 = (v11 - 1) & (v23 + v13);
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v9 == *v14 )
        goto LABEL_6;
      v23 = v29;
    }
    goto LABEL_13;
  }
LABEL_6:
  if ( v14 == (__int64 *)(v12 + 16 * v11)
    || (v16 = (_QWORD *)(*(_QWORD *)(v10 + 24) + 8LL * *((int *)v14 + 2)), (*v16 & 0xFFFFFFFFFFFFFFF8LL) == 0) )
  {
LABEL_13:
    sub_AE6EC0(a1[2], v9);
    return (_QWORD *)sub_2281A50(a1[3], v31, v24, v25, v26, v27);
  }
  v17 = a1[2];
  if ( !*(_QWORD *)(*v16 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    sub_AE6EC0(v17, v9);
    return (_QWORD *)sub_2281A50(a1[3], v31, v24, v25, v26, v27);
  }
  result = sub_AE6EC0(v17, v9);
  if ( (*(_BYTE *)v16 & 4) != 0 )
    return (_QWORD *)sub_2281A50(a1[4], v31, v19, v20, v21, v22);
  return result;
}

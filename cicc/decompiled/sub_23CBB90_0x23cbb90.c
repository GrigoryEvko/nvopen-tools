// Function: sub_23CBB90
// Address: 0x23cbb90
//
unsigned __int64 __fastcall sub_23CBB90(_QWORD *a1, __int64 a2, int a3, int a4)
{
  __int64 v7; // rdx
  unsigned __int64 result; // rax
  unsigned int v9; // esi
  __int64 v10; // r8
  _DWORD *v11; // r11
  int v12; // r15d
  unsigned int v13; // edi
  _DWORD *v14; // rdx
  int v15; // ecx
  unsigned __int64 *v16; // rdx
  int v17; // ecx
  int v18; // ecx
  int v19; // edx
  int v20; // edi
  __int64 v21; // r8
  unsigned int v22; // esi
  int v23; // edx
  int v24; // r10d
  _DWORD *v25; // r9
  int v26; // edx
  int v27; // esi
  __int64 v28; // rdi
  _DWORD *v29; // r8
  unsigned int v30; // ebx
  int v31; // r9d
  int v32; // edx
  unsigned __int64 v33; // [rsp+8h] [rbp-38h]
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]

  v7 = a1[15];
  a1[25] += 40LL;
  result = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[16] >= result + 40 && v7 )
    a1[15] = result + 40;
  else
    result = sub_9D1E70((__int64)(a1 + 15), 40, 40, 3);
  *(_DWORD *)(result + 12) = a3;
  *(_DWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0xFFFFFFFF00000000LL;
  *(_QWORD *)result = &unk_4A16270;
  *(_QWORD *)(result + 24) = -1;
  *(_QWORD *)(result + 32) = a1 + 28;
  v9 = *(_DWORD *)(a2 + 64);
  if ( !v9 )
  {
    ++*(_QWORD *)(a2 + 40);
    goto LABEL_23;
  }
  v10 = *(_QWORD *)(a2 + 48);
  v11 = 0;
  v12 = 1;
  v13 = (v9 - 1) & (37 * a4);
  v14 = (_DWORD *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( *v14 == a4 )
  {
LABEL_6:
    v16 = (unsigned __int64 *)(v14 + 2);
    goto LABEL_7;
  }
  while ( v15 != -1 )
  {
    if ( !v11 && v15 == -2 )
      v11 = v14;
    v13 = (v9 - 1) & (v12 + v13);
    v14 = (_DWORD *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == a4 )
      goto LABEL_6;
    ++v12;
  }
  v17 = *(_DWORD *)(a2 + 56);
  if ( !v11 )
    v11 = v14;
  ++*(_QWORD *)(a2 + 40);
  v18 = v17 + 1;
  if ( 4 * v18 >= 3 * v9 )
  {
LABEL_23:
    v33 = result;
    sub_23CAE70(a2 + 40, 2 * v9);
    v19 = *(_DWORD *)(a2 + 64);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a2 + 48);
      v22 = (v19 - 1) & (37 * a4);
      v18 = *(_DWORD *)(a2 + 56) + 1;
      result = v33;
      v11 = (_DWORD *)(v21 + 16LL * v22);
      v23 = *v11;
      if ( *v11 != a4 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -1 )
        {
          if ( !v25 && v23 == -2 )
            v25 = v11;
          v22 = v20 & (v24 + v22);
          v11 = (_DWORD *)(v21 + 16LL * v22);
          v23 = *v11;
          if ( *v11 == a4 )
            goto LABEL_18;
          ++v24;
        }
        if ( v25 )
          v11 = v25;
      }
      goto LABEL_18;
    }
    goto LABEL_46;
  }
  if ( v9 - *(_DWORD *)(a2 + 60) - v18 <= v9 >> 3 )
  {
    v34 = result;
    sub_23CAE70(a2 + 40, v9);
    v26 = *(_DWORD *)(a2 + 64);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a2 + 48);
      v29 = 0;
      v30 = (v26 - 1) & (37 * a4);
      v31 = 1;
      v18 = *(_DWORD *)(a2 + 56) + 1;
      result = v34;
      v11 = (_DWORD *)(v28 + 16LL * v30);
      v32 = *v11;
      if ( *v11 != a4 )
      {
        while ( v32 != -1 )
        {
          if ( !v29 && v32 == -2 )
            v29 = v11;
          v30 = v27 & (v31 + v30);
          v11 = (_DWORD *)(v28 + 16LL * v30);
          v32 = *v11;
          if ( *v11 == a4 )
            goto LABEL_18;
          ++v31;
        }
        if ( v29 )
          v11 = v29;
      }
      goto LABEL_18;
    }
LABEL_46:
    ++*(_DWORD *)(a2 + 56);
    BUG();
  }
LABEL_18:
  *(_DWORD *)(a2 + 56) = v18;
  if ( *v11 != -1 )
    --*(_DWORD *)(a2 + 60);
  *v11 = a4;
  v16 = (unsigned __int64 *)(v11 + 2);
  *((_QWORD *)v11 + 1) = 0;
LABEL_7:
  *v16 = result;
  return result;
}

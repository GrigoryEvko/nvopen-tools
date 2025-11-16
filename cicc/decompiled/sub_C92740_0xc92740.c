// Function: sub_C92740
// Address: 0xc92740
//
__int64 __fastcall sub_C92740(__int64 a1, const void *a2, size_t a3, int a4)
{
  const void *v4; // r8
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // r15
  int v10; // r10d
  int v11; // r11d
  unsigned int v12; // r13d
  __int64 v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // rsi
  unsigned int v17; // r13d
  int v18; // eax
  const void *v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  int v21; // [rsp+10h] [rbp-40h]
  int v22; // [rsp+14h] [rbp-3Ch]
  int v23; // [rsp+18h] [rbp-38h]

  v4 = a2;
  v7 = *(unsigned int *)(a1 + 8);
  if ( !(_DWORD)v7 )
  {
    sub_C92620(a1, 0x10u);
    v7 = *(unsigned int *)(a1 + 8);
    v4 = a2;
  }
  v8 = v7 - 1;
  v9 = *(_QWORD *)a1;
  v10 = -1;
  v11 = 1;
  v12 = a4 & (v7 - 1);
  v13 = *(_QWORD *)a1 + 8 * v7 + 8;
  v14 = v12;
  v15 = *(_QWORD **)(*(_QWORD *)a1 + 8LL * v12);
  if ( !v15 )
    goto LABEL_4;
  do
  {
    if ( v15 == (_QWORD *)-8LL )
    {
      if ( v10 == -1 )
        v10 = v12;
    }
    else if ( *(_DWORD *)(v13 + 4 * v14) == a4 && a3 == *v15 )
    {
      v21 = v11;
      v20 = v13;
      v22 = v10;
      v23 = v8;
      if ( !a3 )
        return v12;
      v19 = v4;
      v18 = memcmp(v4, (char *)v15 + *(unsigned int *)(a1 + 20), a3);
      v4 = v19;
      v8 = v23;
      v10 = v22;
      v13 = v20;
      v11 = v21;
      if ( !v18 )
        return v12;
    }
    v17 = v11 + v12;
    ++v11;
    v12 = v8 & v17;
    v14 = v12;
    v15 = *(_QWORD **)(v9 + 8LL * v12);
  }
  while ( v15 );
  if ( v10 == -1 )
  {
LABEL_4:
    *(_DWORD *)(v13 + 4 * v14) = a4;
  }
  else
  {
    v12 = v10;
    *(_DWORD *)(v13 + 4LL * v10) = a4;
  }
  return v12;
}

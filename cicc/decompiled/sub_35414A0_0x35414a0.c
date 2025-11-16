// Function: sub_35414A0
// Address: 0x35414a0
//
void __fastcall sub_35414A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 *i; // rax
  __int64 *v11; // r12
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rbx
  __int64 v15; // rsi
  unsigned int v16; // edx
  unsigned int v17; // ecx
  unsigned int v18; // edx
  char v19; // cl
  __int64 *v20; // rdx
  int v21; // r12d
  __int64 *v22; // rax
  unsigned int v23; // eax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx

  v7 = *(unsigned int *)(a1 + 4064);
  if ( !(_DWORD)v7 )
  {
    ++*(_QWORD *)(a1 + 4048);
    goto LABEL_3;
  }
  v9 = *(__int64 **)(a1 + 4056);
  v11 = &v9[2 * *(unsigned int *)(a1 + 4072)];
  if ( v9 == v11 )
    goto LABEL_15;
  v12 = v9;
  while ( 1 )
  {
    v13 = *v12;
    v14 = v12;
    if ( *v12 != -4096 && v13 != -8192 )
      break;
    v12 += 2;
    if ( v11 == v12 )
      goto LABEL_15;
  }
  if ( v11 == v12 )
  {
LABEL_15:
    ++*(_QWORD *)(a1 + 4048);
  }
  else
  {
    do
    {
      v15 = v14[1];
      v14 += 2;
      sub_2E790D0(*(_QWORD *)(a1 + 32), v15, v13, v7, a5, a6);
      if ( v14 == v11 )
        break;
      while ( *v14 == -8192 || *v14 == -4096 )
      {
        v14 += 2;
        if ( v11 == v14 )
          goto LABEL_21;
      }
    }
    while ( v11 != v14 );
LABEL_21:
    LODWORD(v7) = *(_DWORD *)(a1 + 4064);
    ++*(_QWORD *)(a1 + 4048);
    if ( !(_DWORD)v7 )
    {
LABEL_3:
      if ( *(_DWORD *)(a1 + 4068) )
      {
        v8 = *(unsigned int *)(a1 + 4072);
        if ( (unsigned int)v8 <= 0x40 )
        {
          v9 = *(__int64 **)(a1 + 4056);
LABEL_6:
          for ( i = &v9[2 * v8]; i != v9; v9 += 2 )
            *v9 = -4096;
          goto LABEL_8;
        }
        sub_C7D6A0(*(_QWORD *)(a1 + 4056), 16 * v8, 8);
        *(_DWORD *)(a1 + 4072) = 0;
LABEL_45:
        *(_QWORD *)(a1 + 4056) = 0;
LABEL_8:
        *(_QWORD *)(a1 + 4064) = 0;
        goto LABEL_9;
      }
      goto LABEL_9;
    }
    v9 = *(__int64 **)(a1 + 4056);
  }
  v16 = 4 * v7;
  v8 = *(unsigned int *)(a1 + 4072);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v16 = 64;
  if ( v16 >= (unsigned int)v8 )
    goto LABEL_6;
  v17 = v7 - 1;
  if ( v17 )
  {
    _BitScanReverse(&v18, v17);
    v19 = 33 - (v18 ^ 0x1F);
    v20 = v9;
    v21 = 1 << v19;
    if ( 1 << v19 < 64 )
      v21 = 64;
    if ( (_DWORD)v8 == v21 )
    {
      *(_QWORD *)(a1 + 4064) = 0;
      v22 = &v9[2 * v8];
      do
      {
        if ( v20 )
          *v20 = -4096;
        v20 += 2;
      }
      while ( v22 != v20 );
      goto LABEL_9;
    }
  }
  else
  {
    v21 = 64;
  }
  sub_C7D6A0((__int64)v9, 16 * v8, 8);
  v23 = sub_3540050(v21);
  *(_DWORD *)(a1 + 4072) = v23;
  if ( !v23 )
    goto LABEL_45;
  v24 = (_QWORD *)sub_C7D670(16LL * v23, 8);
  v25 = *(unsigned int *)(a1 + 4072);
  *(_QWORD *)(a1 + 4064) = 0;
  *(_QWORD *)(a1 + 4056) = v24;
  for ( j = &v24[2 * v25]; j != v24; v24 += 2 )
  {
    if ( v24 )
      *v24 = -4096;
  }
LABEL_9:
  sub_2F90C70(a1);
}

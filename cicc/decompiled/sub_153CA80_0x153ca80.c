// Function: sub_153CA80
// Address: 0x153ca80
//
bool __fastcall sub_153CA80(__int64 a1, __int64 **a2, __int64 **a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  int v7; // r8d
  unsigned int v8; // r9d
  __int64 *v9; // rdx
  __int64 v10; // r10
  unsigned int v11; // ecx
  unsigned int v12; // r9d
  __int64 *v13; // rdx
  __int64 v14; // r10
  unsigned int v15; // eax
  int v17; // edx
  int v18; // edx
  int v19; // r11d
  int v20; // r11d

  v3 = **a2;
  v4 = **a3;
  if ( v3 == v4 )
    return *((_DWORD *)a2 + 2) > *((_DWORD *)a3 + 2);
  v5 = *(unsigned int *)(*(_QWORD *)a1 + 48LL);
  v6 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !(_DWORD)v5 )
    return 0;
  v7 = v5 - 1;
  v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v3 == *v9 )
  {
LABEL_4:
    v11 = *((_DWORD *)v9 + 2) - 1;
  }
  else
  {
    v18 = 1;
    while ( v10 != -8 )
    {
      v20 = v18 + 1;
      v8 = v7 & (v18 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v3 == *v9 )
        goto LABEL_4;
      v18 = v20;
    }
    v11 = *(_DWORD *)(v6 + 16LL * (unsigned int)v5 + 8) - 1;
  }
  v12 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v13 = (__int64 *)(v6 + 16LL * v12);
  v14 = *v13;
  if ( v4 == *v13 )
  {
LABEL_6:
    v15 = *((_DWORD *)v13 + 2) - 1;
  }
  else
  {
    v17 = 1;
    while ( v14 != -8 )
    {
      v19 = v17 + 1;
      v12 = v7 & (v17 + v12);
      v13 = (__int64 *)(v6 + 16LL * v12);
      v14 = *v13;
      if ( v4 == *v13 )
        goto LABEL_6;
      v17 = v19;
    }
    v15 = *(_DWORD *)(v6 + 16 * v5 + 8) - 1;
  }
  return v15 > v11;
}

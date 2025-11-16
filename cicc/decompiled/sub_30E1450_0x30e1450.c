// Function: sub_30E1450
// Address: 0x30e1450
//
bool __fastcall sub_30E1450(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdi
  unsigned int v6; // r9d
  __int64 *v7; // r8
  unsigned int v8; // r10d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 *v16; // rdx
  int v17; // eax
  int v18; // edx
  int v19; // r11d
  int v20; // r11d

  v4 = *a2;
  v5 = *a3;
  v6 = *(_DWORD *)(*(_QWORD *)a1 + 240LL);
  v7 = *(__int64 **)(*(_QWORD *)a1 + 224LL);
  if ( v6 )
  {
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = &v7[2 * v9];
    v11 = *v10;
    if ( v4 != *v10 )
    {
      v17 = 1;
      while ( v11 != -4096 )
      {
        v19 = v17 + 1;
        v9 = v8 & (v17 + v9);
        v10 = &v7[2 * v9];
        v11 = *v10;
        if ( v4 == *v10 )
          goto LABEL_3;
        v17 = v19;
      }
      v10 = &v7[2 * v6];
    }
LABEL_3:
    v12 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v13 = &v7[2 * v12];
    v14 = *v13;
    if ( v5 == *v13 )
      return *((_DWORD *)v13 + 2) < *((_DWORD *)v10 + 2);
    v18 = 1;
    while ( v14 != -4096 )
    {
      v20 = v18 + 1;
      v12 = v8 & (v18 + v12);
      v13 = &v7[2 * v12];
      v14 = *v13;
      if ( v5 == *v13 )
        return *((_DWORD *)v13 + 2) < *((_DWORD *)v10 + 2);
      v18 = v20;
    }
    v16 = &v7[2 * v6];
    v7 = v10;
  }
  else
  {
    v16 = v7;
  }
  return *((_DWORD *)v16 + 2) < *((_DWORD *)v7 + 2);
}

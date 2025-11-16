// Function: sub_28C9670
// Address: 0x28c9670
//
__int64 __fastcall sub_28C9670(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v4; // r8
  int v5; // r9d
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r10
  int v9; // eax
  __int64 v10; // r8
  int v11; // r9d
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 result; // rax
  __int64 v16; // r8
  int v17; // ecx
  unsigned int v18; // edx
  __int64 v19; // r9
  int v20; // eax
  int v21; // r11d
  int v22; // eax
  int v23; // r11d
  unsigned int v24; // r10d

  v3 = *(_DWORD *)(a1 + 2440);
  v4 = *(_QWORD *)(a1 + 2424);
  if ( v3 )
  {
    v5 = v3 - 1;
    v6 = (v3 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a3 == *v7 )
    {
LABEL_3:
      *v7 = -8192;
      --*(_DWORD *)(a1 + 2432);
      ++*(_DWORD *)(a1 + 2436);
    }
    else
    {
      v20 = 1;
      while ( v8 != -4096 )
      {
        v21 = v20 + 1;
        v6 = v5 & (v20 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a3 == *v7 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  v9 = *(_DWORD *)(a1 + 1656);
  v10 = *(_QWORD *)(a1 + 1640);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( a3 == *v13 )
    {
LABEL_6:
      *v13 = -8192;
      --*(_DWORD *)(a1 + 1648);
      ++*(_DWORD *)(a1 + 1652);
    }
    else
    {
      v22 = 1;
      while ( v14 != -4096 )
      {
        v23 = v22 + 1;
        v12 = v11 & (v22 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( a3 == *v13 )
          goto LABEL_6;
        v22 = v23;
      }
    }
  }
  result = *(unsigned int *)(a1 + 1688);
  v16 = *(_QWORD *)(a1 + 1672);
  if ( (_DWORD)result )
  {
    v17 = result - 1;
    v18 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v16 + 16LL * v18;
    v19 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
    {
LABEL_9:
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(a1 + 1680);
      ++*(_DWORD *)(a1 + 1684);
    }
    else
    {
      result = 1;
      while ( v19 != -4096 )
      {
        v24 = result + 1;
        v18 = v17 & (result + v18);
        result = v16 + 16LL * v18;
        v19 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_9;
        result = v24;
      }
    }
  }
  return result;
}

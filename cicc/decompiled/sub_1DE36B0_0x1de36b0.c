// Function: sub_1DE36B0
// Address: 0x1de36b0
//
__int64 *__fastcall sub_1DE36B0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 *v3; // r11
  __int64 *v4; // r10
  __int64 v5; // r11
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdi
  int v12; // edx
  int v13; // ebx
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 *v16; // rcx
  __int64 *v17; // rsi
  __int64 v18; // rdi
  int v19; // edx
  int v20; // ebx
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 *v24; // rax
  int v25; // ecx
  int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 == a3 )
    return a1;
  v5 = (__int64)a1 + (char *)a3 - (char *)a2;
  v6 = ((char *)a3 - (char *)a1) >> 4;
  v7 = ((char *)a2 - (char *)a1) >> 4;
  if ( v7 == v6 - v7 )
  {
    v24 = a2;
    do
    {
      v25 = *((_DWORD *)v24 + 2);
      v26 = *((_DWORD *)v4 + 2);
      v4 += 2;
      v24 += 2;
      *((_DWORD *)v4 - 2) = v25;
      v27 = *(v24 - 2);
      *((_DWORD *)v24 - 2) = v26;
      v28 = *(v4 - 2);
      *(v4 - 2) = v27;
      *(v24 - 2) = v28;
    }
    while ( a2 != v4 );
    return a2;
  }
  v8 = v6 - v7;
  if ( v7 >= v6 - v7 )
    goto LABEL_12;
  while ( 1 )
  {
    v9 = &v4[2 * v7];
    if ( v8 > 0 )
    {
      v10 = v4;
      v11 = 0;
      do
      {
        v12 = *((_DWORD *)v10 + 2);
        v13 = *((_DWORD *)v9 + 2);
        ++v11;
        v10 += 2;
        v9 += 2;
        *((_DWORD *)v10 - 2) = v13;
        v14 = *(v9 - 2);
        *((_DWORD *)v9 - 2) = v12;
        v15 = *(v10 - 2);
        *(v10 - 2) = v14;
        *(v9 - 2) = v15;
      }
      while ( v8 != v11 );
      v4 += 2 * v8;
    }
    if ( !(v6 % v7) )
      break;
    v8 = v7;
    v7 -= v6 % v7;
    while ( 1 )
    {
      v6 = v8;
      v8 -= v7;
      if ( v7 < v8 )
        break;
LABEL_12:
      v16 = &v4[2 * v6];
      v4 = &v16[-2 * v8];
      if ( v7 > 0 )
      {
        v17 = &v16[-2 * v8];
        v18 = 0;
        do
        {
          v19 = *((_DWORD *)v17 - 2);
          v20 = *((_DWORD *)v16 - 2);
          ++v18;
          v17 -= 2;
          v16 -= 2;
          *((_DWORD *)v17 + 2) = v20;
          v21 = *v16;
          *((_DWORD *)v16 + 2) = v19;
          v22 = *v17;
          *v17 = v21;
          *v16 = v22;
        }
        while ( v7 != v18 );
        v4 -= 2 * v7;
      }
      v7 = v6 % v8;
      if ( !(v6 % v8) )
        return (__int64 *)v5;
    }
  }
  return (__int64 *)v5;
}

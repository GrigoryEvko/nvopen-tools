// Function: sub_325F0C0
// Address: 0x325f0c0
//
__int64 __fastcall sub_325F0C0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r10
  _DWORD *v4; // r9
  unsigned __int64 v5; // rsi
  unsigned int v6; // eax
  int v7; // r8d
  unsigned __int64 v8; // rcx
  _DWORD *v9; // rdi
  _DWORD *v10; // rdx
  __int64 v11; // r11
  _DWORD *v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r11
  __int64 v15; // rcx

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *a2;
  v4 = *(_DWORD **)v2;
  v5 = *(unsigned int *)(v2 + 8);
  v6 = *(_DWORD *)a1 / (unsigned int)v3;
  if ( (unsigned int)v3 <= *(_DWORD *)a1 )
  {
    v7 = 0;
    while ( 1 )
    {
      v8 = v3;
      v9 = v4;
      if ( v5 <= v3 )
        v8 = v5;
      v5 -= v8;
      v4 += v8;
      if ( v7 != *v9 )
        return 0;
      v10 = v9 + 1;
      v11 = 4 * v8 - 4;
      v12 = &v9[v8];
      v13 = v11 >> 4;
      v14 = v11 >> 2;
      if ( v13 > 0 )
      {
        v15 = (__int64)&v9[4 * v13 + 1];
        while ( *v10 == -2 )
        {
          if ( v10[1] != -2 )
          {
            ++v10;
            break;
          }
          if ( v10[2] != -2 )
          {
            v10 += 2;
            break;
          }
          if ( v10[3] != -2 )
          {
            v10 += 3;
            break;
          }
          v10 += 4;
          if ( (_DWORD *)v15 == v10 )
          {
            v14 = v12 - v10;
            goto LABEL_17;
          }
        }
LABEL_13:
        if ( v12 != v10 )
          return 0;
        goto LABEL_14;
      }
LABEL_17:
      if ( v14 != 2 )
      {
        if ( v14 != 3 )
        {
          if ( v14 != 1 || *v10 == -2 )
            goto LABEL_14;
          goto LABEL_21;
        }
        if ( *v10 != -2 )
          goto LABEL_13;
        ++v10;
      }
      if ( *v10 != -2 )
        goto LABEL_13;
      if ( *++v10 == -2 )
        goto LABEL_14;
LABEL_21:
      if ( v12 != v10 )
        return 0;
LABEL_14:
      if ( v6 == ++v7 )
        return 1;
    }
  }
  return 1;
}

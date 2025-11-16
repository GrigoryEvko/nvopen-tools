// Function: sub_2AC3980
// Address: 0x2ac3980
//
char __fastcall sub_2AC3980(__int64 *a1, int *a2)
{
  int v2; // r13d
  char v3; // r12
  char result; // al
  unsigned __int8 *v5; // rsi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // r9
  int v9; // r11d
  unsigned int i; // eax
  __int64 v11; // rdx
  unsigned int v12; // eax
  int v13; // eax
  __int64 v14; // r8
  int v15; // edx
  unsigned int v16; // eax
  unsigned __int8 *v17; // rcx
  int v18; // r9d
  unsigned __int64 v19; // [rsp+8h] [rbp-28h]

  v2 = *a2;
  v3 = *((_BYTE *)a2 + 4);
  v19 = *(_QWORD *)a2;
  result = sub_2AB2DA0(*(_QWORD *)(*a1 + 40), a1[1], *(_QWORD *)a2);
  if ( !result )
  {
    v5 = (unsigned __int8 *)a1[1];
    v6 = *(_QWORD *)(*a1 + 40);
    v7 = *(_DWORD *)(v6 + 152);
    v8 = *(_QWORD *)(v6 + 136);
    if ( v7 )
    {
      v9 = 1;
      for ( i = (v7 - 1) & ((v3 == 0) + 37 * v2 - 1); ; i = (v7 - 1) & v12 )
      {
        v11 = v8 + 40LL * i;
        if ( v2 == *(_DWORD *)v11 && v3 == *(_BYTE *)(v11 + 4) )
          break;
        if ( *(_DWORD *)v11 == -1 && *(_BYTE *)(v11 + 4) )
          goto LABEL_13;
        v12 = v9 + i;
        ++v9;
      }
    }
    else
    {
LABEL_13:
      v11 = v8 + 40LL * v7;
    }
    v13 = *(_DWORD *)(v11 + 32);
    v14 = *(_QWORD *)(v11 + 16);
    if ( v13 )
    {
      v15 = v13 - 1;
      v16 = (v13 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v17 = *(unsigned __int8 **)(v14 + 24LL * v16);
      if ( v5 == v17 )
        return 1;
      v18 = 1;
      while ( v17 != (unsigned __int8 *)-4096LL )
      {
        v16 = v15 & (v18 + v16);
        v17 = *(unsigned __int8 **)(v14 + 24LL * v16);
        if ( v5 == v17 )
          return 1;
        ++v18;
      }
    }
    return sub_2AC3650(v6, v5, v19);
  }
  return result;
}

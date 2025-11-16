// Function: sub_27A2140
// Address: 0x27a2140
//
__int64 __fastcall sub_27A2140(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r8d
  int v5; // eax
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  int v11; // eax
  int v12; // eax
  int v13; // r10d

  v2 = *a2;
  if ( *a2 != 5 )
  {
    v3 = 1;
    if ( (unsigned int)v2 - 12 <= 1 )
      return v3;
    v3 = 0;
    if ( v2 <= 0x15u )
      return v3;
    if ( v2 == 22 )
      return (unsigned int)(*((_DWORD *)a2 + 8) + 3);
    v5 = *(_DWORD *)(a1 + 288);
    v6 = *(_QWORD *)(a1 + 272);
    if ( v5 )
    {
      v7 = v5 - 1;
      v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == (unsigned __int8 *)*v9 )
      {
LABEL_9:
        v11 = *((_DWORD *)v9 + 2);
        if ( v11 )
          return (unsigned int)(v11 + *(_DWORD *)(a1 + 632) + 4);
      }
      else
      {
        v12 = 1;
        while ( v10 != -4096 )
        {
          v13 = v12 + 1;
          v8 = v7 & (v12 + v8);
          v9 = (__int64 *)(v6 + 16LL * v8);
          v10 = *v9;
          if ( a2 == (unsigned __int8 *)*v9 )
            goto LABEL_9;
          v12 = v13;
        }
      }
    }
    return (unsigned int)-1;
  }
  return 2;
}

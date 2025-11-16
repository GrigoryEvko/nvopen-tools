// Function: sub_3247C80
// Address: 0x3247c80
//
unsigned __int8 *__fastcall sub_3247C80(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  int v5; // eax
  int v6; // ecx
  unsigned int v7; // edx
  unsigned __int8 **v8; // rax
  unsigned __int8 *v9; // rdi
  int v10; // eax
  int v11; // r8d
  int v12; // eax
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // edx
  unsigned __int8 *v16; // rdi
  int v18; // eax
  int v19; // r8d

  if ( (unsigned __int8)sub_3247C00((_QWORD *)a1, a2) )
  {
    v3 = *(_QWORD *)(a1 + 216);
    v4 = *(_QWORD *)(v3 + 472);
    v5 = *(_DWORD *)(v3 + 488);
    if ( v5 )
    {
      v6 = v5 - 1;
      v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (unsigned __int8 **)(v4 + 16LL * v7);
      v9 = *v8;
      if ( a2 != *v8 )
      {
        v10 = 1;
        while ( v9 != (unsigned __int8 *)-4096LL )
        {
          v11 = v10 + 1;
          v7 = v6 & (v10 + v7);
          v8 = (unsigned __int8 **)(v4 + 16LL * v7);
          v9 = *v8;
          if ( a2 == *v8 )
            return v8[1];
          v10 = v11;
        }
        return 0;
      }
      return v8[1];
    }
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 256);
    v13 = *(_QWORD *)(a1 + 240);
    if ( v12 )
    {
      v14 = v12 - 1;
      v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (unsigned __int8 **)(v13 + 16LL * v15);
      v16 = *v8;
      if ( a2 != *v8 )
      {
        v18 = 1;
        while ( v16 != (unsigned __int8 *)-4096LL )
        {
          v19 = v18 + 1;
          v15 = v14 & (v18 + v15);
          v8 = (unsigned __int8 **)(v13 + 16LL * v15);
          v16 = *v8;
          if ( a2 == *v8 )
            return v8[1];
          v18 = v19;
        }
        return 0;
      }
      return v8[1];
    }
  }
  return 0;
}

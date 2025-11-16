// Function: sub_2FDF1B0
// Address: 0x2fdf1b0
//
__int64 __fastcall sub_2FDF1B0(_QWORD *a1, unsigned int a2, int a3, unsigned int a4, int a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rcx
  int v10; // esi
  __int64 v11; // r8
  unsigned int v12; // eax
  int v13; // esi
  int v14; // eax
  __int64 v15; // rcx
  int v16; // edx
  __int64 v18; // [rsp+0h] [rbp-8h]

  v5 = a1[13];
  if ( v5
    && (v6 = v5 + 10LL * a2,
        v7 = (unsigned int)*(unsigned __int16 *)(v6 + 6) + a3,
        *(unsigned __int16 *)(v6 + 8) > (unsigned int)v7)
    && (v8 = a1[11],
        v9 = v5 + 10LL * a4,
        v10 = *(_DWORD *)(v8 + 4 * v7),
        v11 = (unsigned int)*(unsigned __int16 *)(v9 + 6) + a5,
        *(unsigned __int16 *)(v9 + 8) > (unsigned int)v11)
    && (v12 = *(_DWORD *)(v8 + 4 * v11), v10 + 1 >= v12) )
  {
    v13 = v10 - v12;
    v14 = v13 + 1;
    if ( v13 != -1 )
    {
      v15 = a1[12];
      v16 = *(_DWORD *)(v15 + 4 * v7);
      if ( v16 )
      {
        if ( v16 == *(_DWORD *)(v15 + 4 * v11) )
          v14 = v13;
      }
    }
    LODWORD(v18) = v14;
    BYTE4(v18) = 1;
  }
  else
  {
    BYTE4(v18) = 0;
  }
  return v18;
}

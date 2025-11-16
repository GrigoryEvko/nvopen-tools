// Function: sub_2786690
// Address: 0x2786690
//
void __fastcall sub_2786690(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v9; // r15
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // rbx
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a2 + 72;
  v7 = *(_QWORD *)(a2 + 80);
  if ( v7 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( v7 )
      {
        v9 = v7 - 24;
        v10 = (unsigned int)(*(_DWORD *)(v7 + 20) + 1);
        v11 = *(_DWORD *)(v7 + 20) + 1;
      }
      else
      {
        v9 = 0;
        v10 = 0;
        v11 = 0;
      }
      if ( v11 < *(_DWORD *)(a3 + 32) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v10) )
        {
          v12 = *(_QWORD *)(v9 + 56);
          v13 = v9 + 48;
          if ( v12 != v13 )
            break;
        }
      }
LABEL_14:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        return;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v12 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 - 16) + 8LL) - 17 <= 1 )
          goto LABEL_8;
        v14 = *(unsigned __int8 *)(v12 - 24) - 29;
        if ( v14 <= 0x2A )
          break;
        if ( *(_BYTE *)(v12 - 24) == 83 )
        {
          v15 = (*(unsigned __int16 *)(v12 - 22) & 0x3Fu) - 1;
          if ( (unsigned int)v15 <= 0xD )
          {
            a4 = dword_4393720;
            if ( dword_4393720[v15] != 42 )
              goto LABEL_13;
          }
        }
LABEL_8:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          goto LABEL_14;
      }
      if ( v14 <= 0x28 )
        goto LABEL_8;
LABEL_13:
      v16[0] = v12 - 24;
      sub_2786050(a1 + 48, v16, a3, (__int64)a4, a5, a6);
      v12 = *(_QWORD *)(v12 + 8);
      if ( v13 == v12 )
        goto LABEL_14;
    }
  }
}

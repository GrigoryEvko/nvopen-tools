// Function: sub_14ECAB0
// Address: 0x14ecab0
//
unsigned __int64 __fastcall sub_14ECAB0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r9d
  __int64 v4; // r11
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // r8
  unsigned int v7; // r14d
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned int v10; // r8d
  unsigned __int64 v11; // rax
  unsigned int v13; // r8d
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  char v18; // cl
  unsigned __int64 v19; // rdx

  v2 = *(_DWORD *)(a1 + 32);
  if ( v2 < a2 )
  {
    v4 = 0;
    if ( v2 )
      v4 = *(_QWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = *(_QWORD *)(a1 + 8);
    v7 = a2 - v2;
    if ( v5 < v6 )
    {
      v8 = v5 + 8;
      v9 = (_QWORD *)(v5 + *(_QWORD *)a1);
      if ( v6 < v5 + 8 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v13 = v6 - v5;
        if ( !v13 )
        {
          *(_DWORD *)(a1 + 32) = 0;
          goto LABEL_15;
        }
        v14 = v13;
        v15 = 0;
        v16 = 0;
        do
        {
          v17 = *((unsigned __int8 *)v9 + v15);
          v18 = 8 * v15++;
          v16 |= v17 << v18;
          *(_QWORD *)(a1 + 24) = v16;
        }
        while ( v13 != v15 );
        v10 = 8 * v13;
        v8 = v5 + v14;
      }
      else
      {
        v10 = 64;
        *(_QWORD *)(a1 + 24) = *v9;
      }
      *(_QWORD *)(a1 + 16) = v8;
      *(_DWORD *)(a1 + 32) = v10;
      if ( v7 <= v10 )
      {
        v11 = *(_QWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 32) = v2 - a2 + v10;
        *(_QWORD *)(a1 + 24) = v11 >> v7;
        return v4 | ((v11 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v2 - (unsigned __int8)a2 + 64))) << v2);
      }
    }
LABEL_15:
    sub_16BD130("Unexpected end of file", 1);
  }
  v19 = *(_QWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 32) = v2 - a2;
  *(_QWORD *)(a1 + 24) = v19 >> a2;
  return v19 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2));
}

// Function: sub_2E21040
// Address: 0x2e21040
//
void __fastcall sub_2E21040(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _DWORD *v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int16 *v12; // r12
  unsigned int v13; // r15d
  _QWORD *v14; // rbx
  int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // r12d
  __int64 v18; // rbx

  v6 = (_DWORD *)(*(_QWORD *)(a1[2] + 32LL) + 4LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
  v7 = (unsigned int)*v6;
  *v6 = 0;
  v8 = *(_QWORD **)(a2 + 104);
  v9 = (_QWORD *)*a1;
  v10 = 24 * v7;
  if ( v8 )
  {
    v11 = v9[1] + v10;
    v12 = (__int16 *)(v9[7] + 2LL * (*(_DWORD *)(v11 + 16) >> 12));
    v13 = *(_DWORD *)(v11 + 16) & 0xFFF;
    v14 = (_QWORD *)(v9[8] + 16LL * *(unsigned __int16 *)(v11 + 20));
    if ( v12 )
    {
      while ( 1 )
      {
        if ( v8 )
        {
          while ( (v8[14] & *v14) == 0 && (v8[15] & v14[1]) == 0 )
          {
            v8 = (_QWORD *)v8[13];
            if ( !v8 )
              goto LABEL_9;
          }
          sub_2E1B600((_DWORD *)(a1[6] + 216LL * v13), a2, (__int64)v8, 27LL * v13, a5);
        }
LABEL_9:
        v15 = *v12;
        v14 += 2;
        ++v12;
        if ( !(_WORD)v15 )
          break;
        v8 = *(_QWORD **)(a2 + 104);
        v13 += v15;
      }
    }
  }
  else
  {
    v16 = *(_DWORD *)(v9[1] + v10 + 16);
    v17 = v16 & 0xFFF;
    v18 = v9[7] + 2LL * (v16 >> 12);
    do
    {
      if ( !v18 )
        break;
      v18 += 2;
      sub_2E1B600((_DWORD *)(a1[6] + 216LL * v17), a2, a2, a4, a5);
      v17 += *(__int16 *)(v18 - 2);
    }
    while ( *(_WORD *)(v18 - 2) );
  }
}

// Function: sub_2E20EE0
// Address: 0x2e20ee0
//
void __fastcall sub_2E20EE0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // r8
  _QWORD *v7; // rdx
  _QWORD *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int16 *v11; // r12
  unsigned int v12; // r15d
  _QWORD *v13; // rbx
  int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // r12d
  __int64 v17; // rbx

  v4 = a3;
  sub_300BF50(a1[2], *(unsigned int *)(a2 + 112), a3);
  v7 = *(_QWORD **)(a2 + 104);
  v8 = (_QWORD *)*a1;
  v9 = 24 * v4;
  if ( v7 )
  {
    v10 = v8[1] + v9;
    v11 = (__int16 *)(v8[7] + 2LL * (*(_DWORD *)(v10 + 16) >> 12));
    v12 = *(_DWORD *)(v10 + 16) & 0xFFF;
    v13 = (_QWORD *)(v8[8] + 16LL * *(unsigned __int16 *)(v10 + 20));
    if ( v11 )
    {
      while ( 1 )
      {
        if ( v7 )
        {
          while ( (v7[14] & *v13) == 0 && (v7[15] & v13[1]) == 0 )
          {
            v7 = (_QWORD *)v7[13];
            if ( !v7 )
              goto LABEL_9;
          }
          sub_2E1D1A0((_DWORD *)(a1[6] + 216LL * v12), a2, (__int64)v7, 27LL * v12, v6);
        }
LABEL_9:
        v14 = *v11;
        v13 += 2;
        ++v11;
        if ( !(_WORD)v14 )
          break;
        v7 = *(_QWORD **)(a2 + 104);
        v12 += v14;
      }
    }
  }
  else
  {
    v15 = *(_DWORD *)(v8[1] + v9 + 16);
    v16 = v15 & 0xFFF;
    v17 = v8[7] + 2LL * (v15 >> 12);
    do
    {
      if ( !v17 )
        break;
      v17 += 2;
      sub_2E1D1A0((_DWORD *)(a1[6] + 216LL * v16), a2, a2, v5, v6);
      v16 += *(__int16 *)(v17 - 2);
    }
    while ( *(_WORD *)(v17 - 2) );
  }
}

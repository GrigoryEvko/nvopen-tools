// Function: sub_1EBC970
// Address: 0x1ebc970
//
__int64 __fastcall sub_1EBC970(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned int v7; // ebx
  unsigned int v8; // esi
  __int16 v9; // ax
  _WORD *v10; // rsi
  __int16 *v11; // rcx
  __int64 v12; // r13
  int v13; // eax
  __int64 i; // r14
  __int64 v15; // r15
  int v16; // esi
  unsigned __int8 v17; // al
  float v18; // xmm0_4
  __int16 v20; // ax
  __int16 *v22; // [rsp+8h] [rbp-58h]
  _BYTE v26[6]; // [rsp+2Ah] [rbp-36h]

  v6 = a1[87];
  if ( !v6 )
    BUG();
  *(_WORD *)&v26[4] = 0;
  v7 = 0;
  v8 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 24LL * a3 + 16);
  v9 = a3 * (v8 & 0xF);
  v10 = (_WORD *)(*(_QWORD *)(v6 + 56) + 2LL * (v8 >> 4));
  v11 = v10 + 1;
  *(_DWORD *)v26 = (unsigned __int16)(*v10 + v9);
LABEL_3:
  v22 = v11;
  while ( v22 )
  {
    v12 = sub_2103840(a1[34], a2, *(unsigned __int16 *)v26, v11, a5, a6);
    v13 = *(_DWORD *)(v12 + 120);
    if ( v13 )
    {
      for ( i = 8LL * (unsigned int)(v13 - 1); i != -8; i -= 8 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(v12 + 112) + i);
        if ( sub_1DB4030(v15, a4, a5) )
        {
          v16 = *(_DWORD *)(v15 + 112);
          if ( v16 >= 0 )
            return 0;
          if ( *(_DWORD *)(a1[115] + 8LL * (v16 & 0x7FFFFFFF)) == 6 )
            return 0;
          v17 = sub_1F5BE30(a1[32]);
          v18 = fmaxf(*(float *)(v15 + 116), *(float *)&v26[2]);
          v7 += v17;
          if ( v7 >= *(_DWORD *)a6 && (v7 != *(_DWORD *)a6 || *(float *)(a6 + 4) <= v18) )
            return 0;
          *(float *)&v26[2] = v18;
        }
      }
    }
    v20 = *v22++;
    v11 = 0;
    if ( !v20 )
      goto LABEL_3;
    *(_WORD *)v26 += v20;
  }
  if ( *(float *)&v26[2] == 0.0 )
    return 0;
  *(_DWORD *)a6 = v7;
  *(_DWORD *)(a6 + 4) = *(_DWORD *)&v26[2];
  return 1;
}

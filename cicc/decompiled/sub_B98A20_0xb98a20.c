// Function: sub_B98A20
// Address: 0xb98a20
//
_QWORD *__fastcall sub_B98A20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 *v4; // rsi
  __int64 v5; // r13
  __int64 v6; // rdi
  int v7; // r14d
  __int64 *v8; // r10
  __int64 v9; // rcx
  __int64 v10; // r9
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r11
  _QWORD *v14; // r13
  int v16; // eax
  int v17; // edx
  __int64 v18; // rbx
  _QWORD *v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26[2]; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v27; // [rsp+18h] [rbp-28h] BYREF

  v26[0] = a1;
  v2 = sub_BD5C60(a1, a2);
  v3 = *(_QWORD *)v2;
  v4 = (__int64 *)*(unsigned int *)(*(_QWORD *)v2 + 592LL);
  v5 = *(_QWORD *)v2 + 568LL;
  if ( !(_DWORD)v4 )
  {
    ++*(_QWORD *)(v3 + 568);
    v27 = 0;
    goto LABEL_27;
  }
  v6 = v26[0];
  v7 = 1;
  v8 = 0;
  v9 = *(_QWORD *)(v3 + 576);
  v10 = v26[0];
  v11 = ((_DWORD)v4 - 1) & ((LODWORD(v26[0]) >> 9) ^ (LODWORD(v26[0]) >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v26[0] != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( !v8 && v13 == -8192 )
        v8 = v12;
      v11 = ((_DWORD)v4 - 1) & (v7 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v26[0] == *v12 )
        goto LABEL_3;
      ++v7;
    }
    if ( !v8 )
      v8 = v12;
    v16 = *(_DWORD *)(v3 + 584);
    ++*(_QWORD *)(v3 + 568);
    v17 = v16 + 1;
    v27 = v8;
    if ( 4 * (v16 + 1) < (unsigned int)(3 * (_DWORD)v4) )
    {
      if ( (int)v4 - *(_DWORD *)(v3 + 588) - v17 > (unsigned int)v4 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(v3 + 584) = v17;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v3 + 588);
        v8[1] = 0;
        v14 = v8 + 1;
        *v8 = v10;
        v6 = v26[0];
        goto LABEL_18;
      }
LABEL_28:
      sub_B98840(v5, (int)v4);
      v4 = v26;
      sub_B927C0(v5, v26, &v27);
      v10 = v26[0];
      v8 = v27;
      v17 = *(_DWORD *)(v3 + 584) + 1;
      goto LABEL_15;
    }
LABEL_27:
    LODWORD(v4) = 2 * (_DWORD)v4;
    goto LABEL_28;
  }
LABEL_3:
  v14 = v12 + 1;
  if ( v12[1] )
    return (_QWORD *)v12[1];
LABEL_18:
  *(_BYTE *)(v6 + 7) |= 8u;
  v18 = v26[0];
  if ( *(_BYTE *)v26[0] > 0x15u )
  {
    v23 = (_QWORD *)sub_22077B0(144);
    v20 = v23;
    if ( !v23 )
      goto LABEL_25;
    v18 = v26[0];
    *v23 = 2;
    v24 = sub_BD5C60(v18, v4);
    v20[2] = 0;
    v20[3] = 0;
    v20[4] = 1;
    v20[1] = v24;
    v25 = v20 + 5;
    do
    {
      if ( v25 )
        *v25 = -4096;
      v25 += 3;
    }
    while ( v20 + 17 != v25 );
    goto LABEL_24;
  }
  v19 = (_QWORD *)sub_22077B0(144);
  v20 = v19;
  if ( v19 )
  {
    *v19 = 1;
    v21 = sub_BD5C60(v18, v4);
    v20[2] = 0;
    v20[3] = 0;
    v20[4] = 1;
    v20[1] = v21;
    v22 = v20 + 5;
    do
    {
      if ( v22 )
        *v22 = -4096;
      v22 += 3;
    }
    while ( v20 + 17 != v22 );
LABEL_24:
    v20[17] = v18;
  }
LABEL_25:
  *v14 = v20;
  return v20;
}

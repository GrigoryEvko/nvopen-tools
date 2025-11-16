// Function: sub_34C97A0
// Address: 0x34c97a0
//
void __fastcall sub_34C97A0(
        _QWORD *a1,
        __m128i a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // r14
  int v10; // r13d
  int v11; // ebx
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r15
  unsigned __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 *v22; // r15
  __int64 v23; // rax
  __int64 v24; // r10
  unsigned __int64 v25; // r11
  _QWORD *v26; // rdx
  _QWORD *v27; // rsi
  unsigned __int64 v28; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v9 = *(_QWORD *)(a1[1] + 32LL);
  v10 = *(_DWORD *)(v9 + 64);
  if ( v10 )
  {
    v11 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = v11 & 0x7FFFFFFF;
        v14 = v11 & 0x7FFFFFFF;
        v15 = *(_QWORD *)(*(_QWORD *)(v9 + 56) + 16 * v14 + 8);
        if ( !v15 )
          goto LABEL_5;
        if ( (*(_BYTE *)(v15 + 4) & 8) == 0 )
          break;
        while ( 1 )
        {
          v15 = *(_QWORD *)(v15 + 32);
          if ( !v15 )
            break;
          if ( (*(_BYTE *)(v15 + 4) & 8) == 0 )
            goto LABEL_8;
        }
        if ( v10 == ++v11 )
          return;
      }
LABEL_8:
      v16 = a1[2];
      v17 = 8 * v14;
      v18 = *(unsigned int *)(v16 + 160);
      if ( (unsigned int)v18 <= v13 )
        break;
      v12 = *(_QWORD *)(*(_QWORD *)(v16 + 152) + 8 * v14);
      if ( !v12 )
        break;
LABEL_4:
      sub_34C9770(a1, v12, a2, a3, a4, a5, a6, a7, a8, a9);
LABEL_5:
      if ( v10 == ++v11 )
        return;
    }
    v19 = v13 + 1;
    v20 = v11 | 0x80000000;
    if ( (unsigned int)v18 < v19 && v19 != v18 )
    {
      if ( v19 >= v18 )
      {
        v24 = *(_QWORD *)(v16 + 168);
        v25 = v19 - v18;
        if ( v19 > (unsigned __int64)*(unsigned int *)(v16 + 164) )
        {
          v28 = v19 - v18;
          v30 = *(_QWORD *)(v16 + 168);
          v32 = a1[2];
          sub_C8D5F0(v16 + 152, (const void *)(v16 + 168), v19, 8u, v16, v20);
          v16 = v32;
          LODWORD(v20) = v11 | 0x80000000;
          v25 = v28;
          v24 = v30;
          v18 = *(unsigned int *)(v32 + 160);
        }
        v21 = *(_QWORD *)(v16 + 152);
        v26 = (_QWORD *)(v21 + 8 * v18);
        v27 = &v26[v25];
        if ( v26 != v27 )
        {
          do
            *v26++ = v24;
          while ( v27 != v26 );
          LODWORD(v18) = *(_DWORD *)(v16 + 160);
          v21 = *(_QWORD *)(v16 + 152);
        }
        *(_DWORD *)(v16 + 160) = v25 + v18;
        goto LABEL_11;
      }
      *(_DWORD *)(v16 + 160) = v19;
    }
    v21 = *(_QWORD *)(v16 + 152);
LABEL_11:
    v22 = (__int64 *)(v21 + v17);
    v29 = (_QWORD *)v16;
    v23 = sub_2E10F30(v20);
    *v22 = v23;
    v31 = v23;
    sub_2E11E80(v29, v23);
    v12 = v31;
    goto LABEL_4;
  }
}

// Function: sub_2CC18B0
// Address: 0x2cc18b0
//
_QWORD *__fastcall sub_2CC18B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v5; // r9
  __int64 v10; // rax
  _QWORD *v11; // r14
  _BYTE *v12; // rsi
  __int64 *v13; // rcx
  __int64 *v14; // r15
  int v15; // esi
  __int64 v16; // r8
  __int64 v17; // rdi
  int v18; // esi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rsi
  _QWORD *v23; // rax
  __int64 *v24; // r15
  __int64 v25; // rdi
  int v27; // eax
  int v28; // r10d
  _BYTE *v29; // rsi
  __int64 v30; // rax
  _QWORD *v31; // [rsp+0h] [rbp-50h]
  _QWORD *v32; // [rsp+8h] [rbp-48h]
  __int64 *v33; // [rsp+8h] [rbp-48h]
  __int64 *v34; // [rsp+8h] [rbp-48h]
  _QWORD *v35; // [rsp+8h] [rbp-48h]
  _QWORD *v36; // [rsp+8h] [rbp-48h]
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1;
  v10 = *(_QWORD *)(a4 + 56);
  *(_QWORD *)(a4 + 136) += 160LL;
  v11 = (_QWORD *)((v10 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a4 + 64) >= (unsigned __int64)(v11 + 20) && v10 )
  {
    *(_QWORD *)(a4 + 56) = v11 + 20;
  }
  else
  {
    v30 = sub_9D1E70(a4 + 56, 160, 160, 3);
    v5 = a1;
    v11 = (_QWORD *)v30;
  }
  memset(v11, 0, 0xA0u);
  v11[9] = 8;
  v11[8] = v11 + 11;
  *((_BYTE *)v11 + 84) = 1;
  v37[0] = v11;
  if ( a2 )
  {
    *v11 = a2;
    v12 = *(_BYTE **)(a2 + 16);
    if ( v12 == *(_BYTE **)(a2 + 24) )
    {
      v35 = v5;
      sub_D4C7F0(a2 + 8, v12, v37);
      v5 = v35;
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v37[0];
        v12 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v12 + 8;
    }
  }
  else
  {
    v29 = *(_BYTE **)(a4 + 40);
    if ( v29 == *(_BYTE **)(a4 + 48) )
    {
      v36 = v5;
      sub_D4C7F0(a4 + 32, v29, v37);
      v5 = v36;
    }
    else
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = v11;
        v29 = *(_BYTE **)(a4 + 40);
      }
      *(_QWORD *)(a4 + 40) = v29 + 8;
    }
  }
  if ( a5 )
  {
    v32 = v5;
    sub_D5B000(a5, v11);
    v5 = v32;
  }
  v13 = (__int64 *)v5[5];
  v14 = (__int64 *)v5[4];
  if ( v14 != v13 )
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)(a4 + 24);
      v16 = *v14;
      v17 = *(_QWORD *)(a4 + 8);
      if ( !v15 )
        goto LABEL_13;
      v18 = v15 - 1;
      v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v16 != *v20 )
      {
        v27 = 1;
        while ( v21 != -4096 )
        {
          v28 = v27 + 1;
          v19 = v18 & (v27 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v16 == *v20 )
            goto LABEL_16;
          v27 = v28;
        }
        goto LABEL_13;
      }
LABEL_16:
      if ( v5 == (_QWORD *)v20[1] )
      {
        v22 = *v14;
        v31 = v5;
        ++v14;
        v33 = v13;
        v23 = sub_2CC1520(a3, v22);
        sub_D4F330(v11, v23[2], a4);
        v13 = v33;
        v5 = v31;
        if ( v33 == v14 )
          break;
      }
      else
      {
LABEL_13:
        if ( v13 == ++v14 )
          break;
      }
    }
  }
  v24 = (__int64 *)v5[1];
  v34 = (__int64 *)v5[2];
  while ( v34 != v24 )
  {
    v25 = *v24++;
    sub_2CC18B0(v25, v11, a3, a4, a5);
  }
  return v11;
}

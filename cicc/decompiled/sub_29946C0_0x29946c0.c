// Function: sub_29946C0
// Address: 0x29946c0
//
void __fastcall sub_29946C0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  unsigned __int64 *v8; // rdx
  __int64 *v9; // r15
  unsigned __int64 v10; // rsi
  unsigned __int64 *v11; // rax
  __int64 *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  _QWORD *v20; // rax
  int v21; // ecx
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  char v24; // r8
  _QWORD *v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // r15
  __int64 *v28; // r14
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-70h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  unsigned __int64 v33; // [rsp+10h] [rbp-60h]
  _QWORD *v34; // [rsp+10h] [rbp-60h]
  _QWORD *v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  unsigned __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int16 v40; // [rsp+38h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 72);
  v8 = *(unsigned __int64 **)(a1 + 64);
  v9 = (__int64 *)v8[v7 - 1];
  *(_DWORD *)(a1 + 72) = v7 - 1;
  v37 = a1 + 144;
  v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_BYTE *)(a1 + 172) )
  {
LABEL_11:
    sub_C8CC70(v37, v10, (__int64)v8, v7, a5, a6);
    if ( !(unsigned __int8)sub_298A9F0(a1, v9) )
      goto LABEL_12;
LABEL_7:
    v12 = *(__int64 **)(a1 + 912);
    if ( v12 )
      sub_2993860(a1, v12, *v9 & 0xFFFFFFFFFFFFFFF8LL, 1);
    *(_QWORD *)(a1 + 912) = v9;
    return;
  }
  v11 = *(unsigned __int64 **)(a1 + 152);
  v7 = *(unsigned int *)(a1 + 164);
  v8 = &v11[v7];
  if ( v11 == v8 )
  {
LABEL_10:
    if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 160) )
      goto LABEL_11;
    *(_DWORD *)(a1 + 164) = v7 + 1;
    *v8 = v10;
    ++*(_QWORD *)(a1 + 144);
  }
  else
  {
    while ( v10 != *v11 )
    {
      if ( v8 == ++v11 )
        goto LABEL_10;
    }
  }
  if ( (unsigned __int8)sub_298A9F0(a1, v9) )
    goto LABEL_7;
LABEL_12:
  v13 = sub_29941A0(a1, 0);
  v14 = v13;
  v15 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_DWORD *)(a1 + 72) && a2 )
  {
    v32 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    sub_B1AEF0(*(_QWORD *)(a1 + 56), v32, v13);
    sub_2993400(a1, v14, v32);
  }
  else
  {
    v32 = sub_2993B60(a1, v13);
  }
  sub_B43C20((__int64)&v39, v14);
  v31 = *(_QWORD *)(a1 + 24);
  v33 = v39;
  v36 = v40;
  v16 = sub_BD2C40(72, 3u);
  if ( v16 )
  {
    v17 = v33;
    v34 = v16;
    sub_B4C9A0((__int64)v16, v15, v32, v31, 3u, v36, v17, v36);
    v16 = v34;
  }
  v35 = v16;
  sub_2988C20(a1, (__int64)v16, v14);
  v19 = *(unsigned int *)(a1 + 664);
  v20 = v35;
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 668) )
  {
    sub_C8D5F0(a1 + 656, (const void *)(a1 + 672), v19 + 1, 8u, v19 + 1, v18);
    v19 = *(unsigned int *)(a1 + 664);
    v20 = v35;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 656) + 8 * v19) = v20;
  ++*(_DWORD *)(a1 + 664);
  sub_2993400(a1, v14, v15);
  sub_B1AEF0(*(_QWORD *)(a1 + 56), v15, v14);
  v21 = *(_DWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 912) = v9;
  if ( !v21 )
    goto LABEL_26;
  while ( !*(_BYTE *)(a1 + 172) )
  {
    if ( sub_C8CA60(v37, a3) )
      goto LABEL_25;
LABEL_30:
    v39 = **(_QWORD **)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72) - 8) & 0xFFFFFFFFFFFFFFF8LL;
    v26 = sub_298A890(a1 + 624, (__int64 *)&v39);
    v27 = (__int64 *)v26[1];
    v28 = &v27[4 * *((unsigned int *)v26 + 6)];
    if ( *((_DWORD *)v26 + 4) && v27 != v28 )
    {
      while ( 1 )
      {
        v29 = *v27;
        if ( *v27 != -4096 && v29 != -8192 )
          break;
        v27 += 4;
        if ( v28 == v27 )
          goto LABEL_31;
      }
      if ( v28 != v27 )
      {
        while ( (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 56), v15, v29) )
        {
          v30 = v27 + 4;
          if ( v28 != v27 + 4 )
          {
            while ( 1 )
            {
              v29 = *v30;
              v27 = v30;
              if ( *v30 != -8192 && v29 != -4096 )
                break;
              v30 += 4;
              if ( v28 == v30 )
                goto LABEL_31;
            }
            if ( v30 != v28 )
              continue;
          }
          goto LABEL_31;
        }
        goto LABEL_25;
      }
    }
LABEL_31:
    sub_2994250(a1, 0, a3);
    if ( !*(_DWORD *)(a1 + 72) )
      goto LABEL_25;
  }
  v22 = *(_QWORD **)(a1 + 152);
  v23 = &v22[*(unsigned int *)(a1 + 164)];
  if ( v22 == v23 )
    goto LABEL_30;
  while ( a3 != *v22 )
  {
    if ( v23 == ++v22 )
      goto LABEL_30;
  }
LABEL_25:
  v9 = *(__int64 **)(a1 + 912);
LABEL_26:
  sub_2993860(a1, v9, v32, 0);
  v24 = sub_22DB400(*(_QWORD **)(a1 + 40), v32);
  v25 = 0;
  if ( v24 )
    v25 = sub_22DDF00(*(_QWORD **)(a1 + 40), v32);
  *(_QWORD *)(a1 + 912) = v25;
}

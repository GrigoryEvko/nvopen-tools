// Function: sub_17F4640
// Address: 0x17f4640
//
unsigned __int64 __fastcall sub_17F4640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r15
  unsigned __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // r9
  unsigned __int64 v11; // r11
  int v12; // eax
  unsigned int v13; // esi
  int v14; // eax
  unsigned __int64 result; // rax
  __int64 v16; // r15
  unsigned __int64 v17; // r9
  unsigned int v18; // r13d
  __int64 *v19; // rsi
  __int64 v20; // rdx
  unsigned int v21; // r13d
  int v22; // eax
  int v23; // eax
  unsigned int v24; // ecx
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  unsigned __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-50h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  unsigned int v38; // [rsp+38h] [rbp-38h]
  unsigned int v39; // [rsp+38h] [rbp-38h]
  __int64 v40; // [rsp+38h] [rbp-38h]

  v28 = a3 & 1;
  v35 = (a3 - 1) / 2;
  if ( a2 >= v35 )
  {
    result = a2;
    v8 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_29;
    v7 = a2;
    goto LABEL_32;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = (__int64 *)(a1 + 16 * (i + 1));
    v10 = *(v8 - 1);
    v9 = *v8;
    v38 = *(_DWORD *)(*v8 + 32);
    if ( v38 > 0x40 )
    {
      v31 = *(v8 - 1);
      v33 = *v8;
      v14 = sub_16A57B0(v9 + 24);
      v9 = v33;
      v11 = -1;
      v10 = v31;
      if ( v38 - v14 <= 0x40 )
        v11 = **(_QWORD **)(v33 + 24);
    }
    else
    {
      v11 = *(_QWORD *)(v9 + 24);
    }
    v39 = *(_DWORD *)(v10 + 32);
    if ( v39 <= 0x40 )
    {
      v6 = *(_QWORD *)(v10 + 24);
    }
    else
    {
      v30 = v11;
      v32 = v9;
      v29 = v10;
      v12 = sub_16A57B0(v10 + 24);
      v9 = v32;
      v11 = v30;
      v13 = v39 - v12;
      v6 = -1;
      if ( v13 <= 0x40 )
        v6 = **(_QWORD **)(v29 + 24);
    }
    if ( v6 > v11 )
    {
      --v7;
      v8 = (__int64 *)(a1 + 8 * v7);
      v9 = *v8;
    }
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v7 >= v35 )
      break;
  }
  if ( !v28 )
  {
LABEL_32:
    if ( (a3 - 2) / 2 == v7 )
    {
      v25 = 2 * v7 + 2;
      v26 = *(_QWORD *)(a1 + 8 * v25 - 8);
      v7 = v25 - 1;
      *v8 = v26;
      v8 = (__int64 *)(a1 + 8 * v7);
    }
  }
  result = v7 - 1;
  v16 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v19 = (__int64 *)(a1 + 8 * v16);
      v20 = *v19;
      v21 = *(_DWORD *)(*v19 + 32);
      if ( v21 <= 0x40 )
        break;
      v40 = *v19;
      v22 = sub_16A57B0(v20 + 24);
      v20 = v40;
      v19 = (__int64 *)(a1 + 8 * v16);
      v17 = -1;
      if ( v21 - v22 > 0x40 )
        goto LABEL_19;
      v18 = *(_DWORD *)(a4 + 32);
      v17 = **(_QWORD **)(v40 + 24);
      if ( v18 <= 0x40 )
      {
LABEL_20:
        result = *(_QWORD *)(a4 + 24);
LABEL_21:
        v8 = (__int64 *)(a1 + 8 * v7);
        if ( result <= v17 )
          goto LABEL_29;
        goto LABEL_22;
      }
LABEL_27:
      v34 = v17;
      v36 = v20;
      v23 = sub_16A57B0(a4 + 24);
      v20 = v36;
      v24 = v18 - v23;
      v17 = v34;
      result = -1;
      if ( v24 > 0x40 )
        goto LABEL_21;
      v8 = (__int64 *)(a1 + 8 * v7);
      result = **(_QWORD **)(a4 + 24);
      if ( result <= v34 )
        goto LABEL_29;
LABEL_22:
      *v8 = v20;
      v7 = v16;
      result = (v16 - 1) / 2;
      if ( a2 >= v16 )
      {
        v8 = v19;
        goto LABEL_29;
      }
      v16 = (v16 - 1) / 2;
    }
    v17 = *(_QWORD *)(v20 + 24);
LABEL_19:
    v18 = *(_DWORD *)(a4 + 32);
    if ( v18 <= 0x40 )
      goto LABEL_20;
    goto LABEL_27;
  }
LABEL_29:
  *v8 = a4;
  return result;
}

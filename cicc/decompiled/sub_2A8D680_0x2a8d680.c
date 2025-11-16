// Function: sub_2A8D680
// Address: 0x2a8d680
//
__int64 __fastcall sub_2A8D680(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // r8
  __int64 v4; // rdx
  _QWORD *v5; // rdi
  signed __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r9
  int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // r9
  int v14; // ecx
  __int64 v15; // rdx
  __int64 v16; // r9
  int v17; // ecx
  _BYTE *v18; // rdx
  __int64 v19; // r9
  int v20; // ecx
  _BYTE *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // r12d
  _QWORD *v26; // rax
  __int64 result; // rax
  _BYTE *v28; // rdx
  __int64 v29; // rsi
  int v30; // ecx
  _QWORD *v31; // rcx
  _BYTE *v32; // rax
  int v33; // esi
  _BYTE *v34; // rax
  _BYTE *v35; // rdx
  __int64 v36; // rsi
  int v37; // ecx
  _BYTE *v38; // rdx
  __int64 v39; // rsi
  int v40; // ecx
  _QWORD v41[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *(_QWORD **)a2;
  v4 = 24LL * *(unsigned int *)(a2 + 8);
  v5 = (_QWORD *)(*(_QWORD *)a2 + v4);
  v6 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 3);
  if ( !(v6 >> 2) )
  {
    v7 = *(_QWORD **)a2;
LABEL_32:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_40;
        goto LABEL_35;
      }
      v35 = (_BYTE *)*v7;
      if ( *(_BYTE *)*v7 != 61 )
        v35 = (_BYTE *)*((_QWORD *)v35 - 8);
      v36 = *((_QWORD *)v35 + 1);
      v37 = *(unsigned __int8 *)(v36 + 8);
      if ( (unsigned int)(v37 - 17) <= 1 )
        LOBYTE(v37) = *(_BYTE *)(**(_QWORD **)(v36 + 16) + 8LL);
      if ( (_BYTE)v37 == 14 )
        goto LABEL_24;
      v7 += 3;
    }
    v38 = (_BYTE *)*v7;
    if ( *(_BYTE *)*v7 != 61 )
      v38 = (_BYTE *)*((_QWORD *)v38 - 8);
    v39 = *((_QWORD *)v38 + 1);
    v40 = *(unsigned __int8 *)(v39 + 8);
    if ( (unsigned int)(v40 - 17) <= 1 )
      LOBYTE(v40) = *(_BYTE *)(**(_QWORD **)(v39 + 16) + 8LL);
    if ( (_BYTE)v40 == 14 )
      goto LABEL_24;
    v7 += 3;
LABEL_35:
    v28 = (_BYTE *)*v7;
    if ( *(_BYTE *)*v7 != 61 )
      v28 = (_BYTE *)*((_QWORD *)v28 - 8);
    v29 = *((_QWORD *)v28 + 1);
    v30 = *(unsigned __int8 *)(v29 + 8);
    if ( (unsigned int)(v30 - 17) <= 1 )
      LOBYTE(v30) = *(_BYTE *)(**(_QWORD **)(v29 + 16) + 8LL);
    if ( (_BYTE)v30 != 14 )
      goto LABEL_40;
    goto LABEL_24;
  }
  v7 = *(_QWORD **)a2;
  v8 = &v3[12 * (v6 >> 2)];
  while ( 1 )
  {
    v18 = (_BYTE *)*v7;
    if ( *(_BYTE *)*v7 != 61 )
      v18 = (_BYTE *)*((_QWORD *)v18 - 8);
    v19 = *((_QWORD *)v18 + 1);
    v20 = *(unsigned __int8 *)(v19 + 8);
    if ( (unsigned int)(v20 - 17) <= 1 )
      LOBYTE(v20) = *(_BYTE *)(**(_QWORD **)(v19 + 16) + 8LL);
    if ( (_BYTE)v20 == 14 )
      break;
    v9 = v7[3];
    if ( *(_BYTE *)v9 != 61 )
      v9 = *(_QWORD *)(v9 - 64);
    v10 = *(_QWORD *)(v9 + 8);
    v11 = *(unsigned __int8 *)(v10 + 8);
    if ( (unsigned int)(v11 - 17) <= 1 )
      LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
    if ( (_BYTE)v11 == 14 )
    {
      v7 += 3;
      break;
    }
    v12 = v7[6];
    if ( *(_BYTE *)v12 != 61 )
      v12 = *(_QWORD *)(v12 - 64);
    v13 = *(_QWORD *)(v12 + 8);
    v14 = *(unsigned __int8 *)(v13 + 8);
    if ( (unsigned int)(v14 - 17) <= 1 )
      LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
    if ( (_BYTE)v14 == 14 )
    {
      v7 += 6;
      break;
    }
    v15 = v7[9];
    if ( *(_BYTE *)v15 != 61 )
      v15 = *(_QWORD *)(v15 - 64);
    v16 = *(_QWORD *)(v15 + 8);
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( (unsigned int)(v17 - 17) <= 1 )
      LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
    if ( (_BYTE)v17 == 14 )
    {
      v7 += 9;
      break;
    }
    v7 += 12;
    if ( v8 == v7 )
    {
      v6 = 0xAAAAAAAAAAAAAAABLL * (v5 - v7);
      goto LABEL_32;
    }
  }
LABEL_24:
  if ( v5 != v7 )
  {
    v21 = (_BYTE *)*v3;
    if ( *(_BYTE *)*v3 != 61 )
      v21 = (_BYTE *)*((_QWORD *)v21 - 8);
    v22 = *((_QWORD *)v21 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
      v22 = **(_QWORD **)(v22 + 16);
    v23 = sub_9208B0(a1[6], v22);
    v41[1] = v24;
    v41[0] = v23;
    v25 = sub_CA1930(v41);
    v26 = (_QWORD *)sub_B2BE50(*a1);
    return sub_BCD140(v26, v25);
  }
LABEL_40:
  if ( v3 == v5 )
  {
LABEL_48:
    v34 = (_BYTE *)*v3;
    if ( *(_BYTE *)*v3 != 61 )
      v34 = (_BYTE *)*((_QWORD *)v34 - 8);
    result = *((_QWORD *)v34 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(result + 8) - 17 <= 1 )
      return **(_QWORD **)(result + 16);
  }
  else
  {
    v31 = v3;
    while ( 1 )
    {
      v32 = (_BYTE *)*v31;
      if ( *(_BYTE *)*v31 != 61 )
        v32 = (_BYTE *)*((_QWORD *)v32 - 8);
      result = *((_QWORD *)v32 + 1);
      v33 = *(unsigned __int8 *)(result + 8);
      if ( (unsigned int)(v33 - 17) <= 1 )
      {
        result = **(_QWORD **)(result + 16);
        LOBYTE(v33) = *(_BYTE *)(result + 8);
      }
      if ( (_BYTE)v33 == 12 )
        break;
      v31 += 3;
      if ( v5 == v31 )
        goto LABEL_48;
    }
  }
  return result;
}

// Function: sub_22DE410
// Address: 0x22de410
//
unsigned __int64 __fastcall sub_22DE410(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  int v5; // edx
  __int64 v6; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rsi
  _QWORD *v10; // r8
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rax
  __int64 v17; // r13
  unsigned int v18; // r14d
  _QWORD *v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // r8
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  char v25; // dl
  unsigned __int64 v26; // rcx
  _QWORD *v27; // r14
  __int64 v28; // rbx
  unsigned int v29; // r12d
  int v30; // eax
  unsigned int v31; // esi
  __int64 v32; // [rsp+10h] [rbp-90h]
  _QWORD *v33; // [rsp+18h] [rbp-88h]
  unsigned __int64 v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  unsigned __int64 v36; // [rsp+28h] [rbp-78h]
  unsigned __int64 v37; // [rsp+30h] [rbp-70h]
  unsigned int v38; // [rsp+38h] [rbp-68h]
  _QWORD *v39; // [rsp+38h] [rbp-68h]
  __m128i v40[2]; // [rsp+40h] [rbp-60h] BYREF
  char v41; // [rsp+60h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 104);
  while ( 1 )
  {
    v36 = *(_QWORD *)(v1 - 40);
    v37 = v36 & 0xFFFFFFFFFFFFFFF9LL;
    if ( *(_BYTE *)(v1 - 8) )
      goto LABEL_3;
    v34 = v37 | (*(_QWORD *)v36 >> 1) & 2LL;
    v39 = (_QWORD *)((*(_QWORD *)v36 & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v26 = *v39 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = (_QWORD *)v26;
    if ( v39 == (_QWORD *)v26 )
    {
      v28 = 0;
    }
    else
    {
      if ( !v26 )
LABEL_26:
        BUG();
      v28 = v26 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 >= 0xB )
        v28 = 0;
    }
    if ( ((*(_QWORD *)v36 >> 1) & 2) != 0 )
    {
      if ( *(_QWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 8)
                                                                       + 32LL) )
        v34 = v36 & 0xFFFFFFFFFFFFFFF9LL | 4;
      v30 = 0;
    }
    else
    {
      v32 = v1;
      v29 = 0;
      do
      {
        v35 = v29;
        if ( v39 == v27 )
          goto LABEL_51;
        if ( !v27 )
          goto LABEL_26;
        if ( (unsigned int)*((unsigned __int8 *)v27 - 24) - 30 > 0xA )
        {
LABEL_51:
          v30 = 0;
          if ( !v29 )
          {
LABEL_52:
            v1 = v32;
            goto LABEL_50;
          }
        }
        else
        {
          v30 = sub_B46E30((__int64)(v27 - 3));
          if ( v29 == v30 )
            goto LABEL_52;
        }
        v31 = v29++;
      }
      while ( *(_QWORD *)(*(_QWORD *)((v34 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) == sub_B46EC0(v28, v31) );
      v1 = v32;
      v30 = v35;
    }
LABEL_50:
    *(_QWORD *)(v1 - 24) = v28;
    *(_DWORD *)(v1 - 16) = v30;
    *(_QWORD *)(v1 - 32) = v34;
    *(_BYTE *)(v1 - 8) = 1;
LABEL_3:
    v2 = v36 & 0xFFFFFFFFFFFFFFF9LL;
    if ( (*(_QWORD *)v36 & 4) != 0 )
      v2 = v37 | 4;
    v3 = *(_QWORD *)v36 & 0xFFFFFFFFFFFFFFF8LL;
    v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == v3 + 48 )
      goto LABEL_32;
    if ( !v4 )
      goto LABEL_26;
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_32:
      v5 = 0;
    else
      v5 = sub_B46E30(v4 - 24);
    v6 = *(_QWORD *)(v1 - 32);
    result = (v6 >> 1) & 3;
    if ( ((v6 >> 1) & 3) != 0 )
    {
      if ( (_DWORD)result != ((v2 >> 1) & 3) )
      {
        v8 = v6 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v1 - 32) = *(_QWORD *)(v1 - 32) & 0xFFFFFFFFFFFFFFF9LL | 4;
        v9 = *(_QWORD *)(v8 + 32);
        v10 = *(_QWORD **)(v8 + 8);
        goto LABEL_12;
      }
      goto LABEL_33;
    }
    result = *(unsigned int *)(v1 - 16);
    v38 = result;
    if ( (_DWORD)result != v5 )
      break;
LABEL_33:
    *(_QWORD *)(a1 + 104) -= 40LL;
    v1 = *(_QWORD *)(a1 + 104);
    if ( v1 == *(_QWORD *)(a1 + 96) )
      return result;
  }
  v17 = *(_QWORD *)(v1 - 24);
  v18 = *(_DWORD *)(v1 - 16);
  v19 = (_QWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
  while ( 2 )
  {
    *(_DWORD *)(v1 - 16) = ++v18;
    v22 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
    v23 = *(_QWORD *)(v22 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v23 != v22 + 48 )
    {
      if ( !v23 )
        goto LABEL_26;
      if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 <= 0xA )
      {
        if ( v18 == (unsigned int)sub_B46E30(v23 - 24) )
          goto LABEL_28;
LABEL_23:
        v20 = sub_B46EC0(v17, v18);
        v21 = (_QWORD *)v19[1];
        if ( v21[4] != v20 )
          goto LABEL_29;
        continue;
      }
    }
    break;
  }
  if ( v18 )
    goto LABEL_23;
LABEL_28:
  v21 = (_QWORD *)v19[1];
LABEL_29:
  v33 = v21;
  v24 = sub_B46EC0(v17, v38);
  v10 = v33;
  v9 = v24;
LABEL_12:
  v13 = sub_22DE030(v10, v9);
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_30;
  v16 = *(_QWORD **)(a1 + 8);
  v12 = *(unsigned int *)(a1 + 20);
  v11 = &v16[v12];
  if ( v16 != v11 )
  {
    while ( v13 != (_QWORD *)*v16 )
    {
      if ( v11 == ++v16 )
        goto LABEL_16;
    }
    goto LABEL_3;
  }
LABEL_16:
  if ( (unsigned int)v12 < *(_DWORD *)(a1 + 16) )
  {
    *(_DWORD *)(a1 + 20) = v12 + 1;
    *v11 = v13;
    ++*(_QWORD *)a1;
  }
  else
  {
LABEL_30:
    sub_C8CC70(a1, (__int64)v13, (__int64)v11, v12, v14, v15);
    if ( !v25 )
      goto LABEL_3;
  }
  v40[0].m128i_i64[0] = (__int64)v13;
  v41 = 0;
  return sub_22DD390((unsigned __int64 *)(a1 + 96), v40);
}

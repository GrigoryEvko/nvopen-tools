// Function: sub_22E7560
// Address: 0x22e7560
//
unsigned __int64 __fastcall sub_22E7560(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 v2; // rdx
  int v3; // eax
  unsigned int v4; // r15d
  _QWORD *v5; // rbx
  __int64 v6; // r13
  unsigned int v7; // r14d
  unsigned int v8; // esi
  __int64 v9; // rax
  _QWORD *v10; // r8
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 *v20; // rax
  unsigned __int64 result; // rax
  char v22; // dl
  unsigned __int64 v23; // rbx
  __int64 v24; // r13
  unsigned int v25; // r15d
  unsigned int v26; // esi
  unsigned int v27; // [rsp+4h] [rbp-7Ch]
  _QWORD *v28; // [rsp+10h] [rbp-70h]
  _QWORD *v29; // [rsp+10h] [rbp-70h]
  _QWORD *v30; // [rsp+18h] [rbp-68h]
  __m128i v31[2]; // [rsp+20h] [rbp-60h] BYREF
  char v32; // [rsp+40h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 104);
  while ( 2 )
  {
    v30 = *(_QWORD **)(v1 - 40);
    if ( *(_BYTE *)(v1 - 8) )
      goto LABEL_3;
    v29 = (_QWORD *)((**(_QWORD **)(v1 - 40) & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v23 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29 == (_QWORD *)v23 )
    {
      v24 = 0;
    }
    else
    {
      if ( !v23 )
LABEL_14:
        BUG();
      v24 = v23 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 >= 0xB )
        v24 = 0;
    }
    v25 = 0;
    do
    {
      v27 = v25;
      if ( v29 != (_QWORD *)v23 )
      {
        if ( !v23 )
          goto LABEL_14;
        if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 <= 0xA )
        {
          if ( (unsigned int)sub_B46E30(v23 - 24) == v25 )
            break;
          goto LABEL_37;
        }
      }
      if ( !v25 )
        break;
LABEL_37:
      v26 = v25++;
    }
    while ( *(_QWORD *)(v30[1] + 32LL) == sub_B46EC0(v24, v26) );
    *(_QWORD *)(v1 - 24) = v24;
    *(_BYTE *)(v1 - 8) = 1;
    *(_QWORD *)(v1 - 32) = v30;
    *(_DWORD *)(v1 - 16) = v27;
LABEL_3:
    v2 = *(_QWORD *)((*v30 & 0xFFFFFFFFFFFFFFF8LL) + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v2 == (*v30 & 0xFFFFFFFFFFFFFFF8LL) + 48 )
      goto LABEL_29;
    if ( !v2 )
      goto LABEL_14;
    if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_29:
      v3 = 0;
    else
      v3 = sub_B46E30(v2 - 24);
    v4 = *(_DWORD *)(v1 - 16);
    v5 = *(_QWORD **)(v1 - 32);
    if ( v4 == v3 && v30 == v5 )
    {
      result = a1;
      *(_QWORD *)(a1 + 104) -= 40LL;
      v1 = *(_QWORD *)(a1 + 104);
      if ( v1 == *(_QWORD *)(a1 + 96) )
        return result;
      continue;
    }
    break;
  }
  v6 = *(_QWORD *)(v1 - 24);
  v7 = v4 + 1;
  while ( 2 )
  {
    *(_DWORD *)(v1 - 16) = v7;
    v11 = *v5 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 != v11 + 48 )
    {
      if ( !v12 )
        goto LABEL_14;
      if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 <= 0xA )
      {
        if ( v7 == (unsigned int)sub_B46E30(v12 - 24) )
          goto LABEL_16;
LABEL_11:
        v8 = v7++;
        v9 = sub_B46EC0(v6, v8);
        v10 = (_QWORD *)v5[1];
        if ( v10[4] != v9 )
          goto LABEL_17;
        continue;
      }
    }
    break;
  }
  if ( v7 )
    goto LABEL_11;
LABEL_16:
  v10 = (_QWORD *)v5[1];
LABEL_17:
  v28 = v10;
  v13 = sub_B46EC0(v6, v4);
  v14 = sub_22DDF00(v28, v13);
  v18 = a1;
  v19 = (__int64)v14;
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_24;
  v20 = *(__int64 **)(a1 + 8);
  v18 = *(unsigned int *)(a1 + 20);
  v15 = &v20[v18];
  if ( v20 != v15 )
  {
    while ( v19 != *v20 )
    {
      if ( v15 == ++v20 )
        goto LABEL_21;
    }
    goto LABEL_3;
  }
LABEL_21:
  if ( (unsigned int)v18 < *(_DWORD *)(a1 + 16) )
  {
    *(_DWORD *)(a1 + 20) = v18 + 1;
    *v15 = v19;
    ++*(_QWORD *)a1;
  }
  else
  {
LABEL_24:
    sub_C8CC70(a1, v19, (__int64)v15, v18, v16, v17);
    if ( !v22 )
      goto LABEL_3;
  }
  v31[0].m128i_i64[0] = v19;
  v32 = 0;
  return sub_22E6150((unsigned __int64 *)(a1 + 96), v31);
}

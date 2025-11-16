// Function: sub_32A0D60
// Address: 0x32a0d60
//
__int64 __fastcall sub_32A0D60(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rdi
  int v7; // esi
  __int64 v8; // r8
  int v9; // r9d
  __int64 *v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rbx
  __int64 v13; // r12
  int v14; // r14d
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v20; // r8d
  __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r8
  int v24; // r10d
  __int64 v25; // r9
  int v26; // edx
  __int64 v27; // r11
  unsigned __int16 *v28; // rax
  __int64 *v29; // [rsp-30h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( *((_DWORD *)a4 + 8) != v4 )
    goto LABEL_2;
  v21 = *(__int64 **)(a1 + 40);
  v22 = a4[5];
  v23 = *v21;
  v24 = *((_DWORD *)v21 + 2);
  v25 = v21[5];
  if ( v22 )
  {
    if ( v23 != v22 || v24 != *((_DWORD *)a4 + 12) )
    {
      v26 = *((_DWORD *)v21 + 12);
      goto LABEL_21;
    }
  }
  else if ( !v23 )
  {
    goto LABEL_41;
  }
  v27 = a4[7];
  if ( !v27 )
  {
    if ( !v25 )
      goto LABEL_2;
    goto LABEL_26;
  }
  v26 = *((_DWORD *)v21 + 12);
  if ( v27 == v25 )
  {
    if ( v26 == *((_DWORD *)a4 + 16) )
      goto LABEL_26;
    if ( !v22 )
      goto LABEL_24;
    goto LABEL_21;
  }
  if ( v22 )
  {
LABEL_21:
    if ( v22 != v25 || *((_DWORD *)a4 + 12) != v26 )
      goto LABEL_2;
    goto LABEL_23;
  }
LABEL_41:
  if ( !v25 )
    goto LABEL_2;
LABEL_23:
  v27 = a4[7];
  if ( v27 )
  {
LABEL_24:
    if ( v23 != v27 || v24 != *((_DWORD *)a4 + 16) )
      goto LABEL_2;
    goto LABEL_26;
  }
  if ( !v23 )
    goto LABEL_2;
LABEL_26:
  v20 = *((unsigned __int8 *)a4 + 76);
  if ( !(_BYTE)v20 )
    return 1;
  if ( *((_DWORD *)a4 + 18) == ((_DWORD)a4[9] & *(_DWORD *)(a1 + 28)) )
    return v20;
LABEL_2:
  if ( (unsigned int)(v4 - 205) > 1 )
    return 0;
  v5 = *(_QWORD *)(a1 + 40);
  if ( *(_DWORD *)(*(_QWORD *)v5 + 24LL) != 208 )
    return 0;
  v6 = *(_QWORD *)(v5 + 40);
  v7 = *(_DWORD *)(v5 + 48);
  v8 = *(_QWORD *)(v5 + 80);
  v9 = *(_DWORD *)(v5 + 88);
  v10 = *(__int64 **)(*(_QWORD *)v5 + 40LL);
  v11 = *((_DWORD *)v10 + 2);
  v12 = *v10;
  v13 = v10[5];
  v14 = *((_DWORD *)v10 + 12);
  v15 = v10[10];
  if ( v8 == v13 && v6 == *v10 && v7 == v11 && v9 == v14 )
    goto LABEL_6;
  if ( v8 != v12 || v6 != v13 || v7 != v14 || v9 != v11 )
    return 0;
  if ( v6 == v12 && v7 == v11 )
  {
LABEL_6:
    v16 = *(_DWORD *)(v15 + 96);
  }
  else
  {
    v29 = a4;
    v28 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v11);
    v16 = sub_33CBD40(*(unsigned int *)(v15 + 96), *v28, *((_QWORD *)v28 + 1));
    a4 = v29;
  }
  if ( (unsigned int)(v16 - 20) <= 1 )
  {
    v17 = *a4;
    if ( *a4 )
    {
      if ( v12 != v17 || v11 != *((_DWORD *)a4 + 2) )
        goto LABEL_10;
    }
    else if ( !v12 )
    {
      return 0;
    }
    v18 = a4[2];
    if ( !v18 )
      return v13 != 0;
    if ( v13 == v18 )
    {
      if ( v14 == *((_DWORD *)a4 + 6) )
        return 1;
      if ( !v17 )
        return v12 == v18 && v11 == *((_DWORD *)a4 + 6);
    }
    else if ( !v17 )
    {
      if ( v13 )
        return v12 == v18 && v11 == *((_DWORD *)a4 + 6);
      return 0;
    }
LABEL_10:
    if ( v13 != v17 || v14 != *((_DWORD *)a4 + 2) )
      return 0;
    v18 = a4[2];
    if ( v18 )
      return v12 == v18 && v11 == *((_DWORD *)a4 + 6);
    return v12 != 0;
  }
  return 0;
}

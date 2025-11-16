// Function: sub_FC95E0
// Address: 0xfc95e0
//
__int64 *__fastcall sub_FC95E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // r13
  unsigned int v22; // esi
  __int64 v23; // r8
  __int64 v24; // rcx
  int v25; // r11d
  __int64 *v26; // r10
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // r12
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  int v33; // eax
  __int64 v34; // rax
  int v35; // edi
  int v36; // ebx
  unsigned __int8 *v37; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int8 *v38; // [rsp+28h] [rbp-38h] BYREF
  __int64 *v39; // [rsp+30h] [rbp-30h] BYREF
  __int64 v40; // [rsp+38h] [rbp-28h]

  v6 = a2;
  v8 = *(unsigned int *)(a1 + 16);
  v37 = (unsigned __int8 *)a2;
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16 * v8);
  if ( *(_BYTE *)(v9 + 64) )
  {
    a3 = *(unsigned int *)(v9 + 56);
    v13 = *(_QWORD *)(v9 + 40);
    if ( (_DWORD)a3 )
    {
      a4 = ((_DWORD)a3 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v14 = (__int64 *)(v13 + 16 * a4);
      a5 = *v14;
      if ( v6 == *v14 )
      {
LABEL_13:
        a3 = v13 + 16 * a3;
        if ( v14 != (__int64 *)a3 )
        {
          v15 = v14[1];
          LOBYTE(v40) = 1;
          return (__int64 *)v15;
        }
      }
      else
      {
        v33 = 1;
        while ( a5 != -4096 )
        {
          a6 = (unsigned int)(v33 + 1);
          a4 = ((_DWORD)a3 - 1) & (unsigned int)(v33 + a4);
          v14 = (__int64 *)(v13 + 16LL * (unsigned int)a4);
          a5 = *v14;
          if ( v6 == *v14 )
            goto LABEL_13;
          v33 = a6;
        }
      }
    }
  }
  if ( !*(_BYTE *)v6 || (*(_BYTE *)a1 & 1) != 0 )
  {
    v39 = (__int64 *)v6;
    LOBYTE(v40) = 1;
    return v39;
  }
  if ( *(_BYTE *)v6 == 1 )
  {
    v11 = *(_QWORD *)(v6 + 136);
    v12 = sub_FC8800(a1, v11, a3, a4, a5, a6);
    if ( v12 == *(_QWORD *)(v6 + 136) )
    {
      v12 = v6;
    }
    else if ( v12 )
    {
      v12 = (__int64)sub_B98A20(v12, v11);
    }
LABEL_10:
    v39 = (__int64 *)v12;
    LOBYTE(v40) = 1;
    return v39;
  }
  v16 = *(_QWORD *)(a1 + 360);
  if ( !v16 )
    goto LABEL_31;
  if ( *(_BYTE *)(v16 + 28) )
  {
    v17 = *(_QWORD **)(v16 + 8);
    v18 = &v17[*(unsigned int *)(v16 + 20)];
    if ( v17 == v18 )
      goto LABEL_31;
    while ( v6 != *v17 )
    {
      if ( v18 == ++v17 )
        goto LABEL_31;
    }
    v38 = (unsigned __int8 *)v6;
    goto LABEL_22;
  }
  if ( !sub_C8CA60(v16, v6) )
  {
LABEL_31:
    LOBYTE(v40) = 0;
    return v39;
  }
  v6 = (__int64)v37;
  v38 = v37;
  if ( v37 )
LABEL_22:
    sub_B96E90((__int64)&v38, v6, 1);
  v19 = *(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 16);
  v20 = *(_QWORD *)v19;
  v21 = *(_QWORD *)v19 + 32LL;
  if ( !*(_BYTE *)(*(_QWORD *)v19 + 64LL) )
  {
    *(_QWORD *)(v20 + 40) = 0;
    v34 = 1;
    *(_QWORD *)(v20 + 48) = 0;
    *(_DWORD *)(v20 + 56) = 0;
    *(_BYTE *)(v20 + 64) = 1;
LABEL_41:
    v39 = 0;
    v22 = 0;
    *(_QWORD *)(v20 + 32) = v34;
    goto LABEL_42;
  }
  v22 = *(_DWORD *)(v20 + 56);
  v23 = *(_QWORD *)(v20 + 40);
  if ( !v22 )
  {
    v34 = *(_QWORD *)(v20 + 32) + 1LL;
    goto LABEL_41;
  }
  v24 = (__int64)v37;
  v25 = 1;
  v26 = 0;
  v27 = (v22 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
  v28 = (__int64 *)(v23 + 16LL * v27);
  v29 = *v28;
  if ( v37 == (unsigned __int8 *)*v28 )
  {
LABEL_26:
    v30 = v28 + 1;
    v31 = v28[1];
    if ( v28 + 1 != (__int64 *)&v38 )
    {
      if ( v31 )
        sub_B91220((__int64)(v28 + 1), v31);
LABEL_29:
      v32 = v38;
      *v30 = (__int64)v38;
      if ( !v32 )
      {
        v39 = 0;
        LOBYTE(v40) = 1;
        return v39;
      }
      sub_B976B0((__int64)&v38, v32, (__int64)v30);
      v38 = 0;
      v12 = *v30;
      goto LABEL_10;
    }
    goto LABEL_48;
  }
  while ( v29 != -4096 )
  {
    if ( !v26 && v29 == -8192 )
      v26 = v28;
    v27 = (v22 - 1) & (v25 + v27);
    v28 = (__int64 *)(v23 + 16LL * v27);
    v29 = *v28;
    if ( v37 == (unsigned __int8 *)*v28 )
      goto LABEL_26;
    ++v25;
  }
  if ( v26 )
    v28 = v26;
  v39 = v28;
  v36 = *(_DWORD *)(v20 + 48);
  ++*(_QWORD *)(v20 + 32);
  v35 = v36 + 1;
  if ( 4 * (v36 + 1) < 3 * v22 )
  {
    if ( v22 - *(_DWORD *)(v20 + 52) - v35 > v22 >> 3 )
      goto LABEL_44;
    goto LABEL_43;
  }
LABEL_42:
  v22 *= 2;
LABEL_43:
  sub_FC7EB0(v21, v22);
  sub_FC7DF0(v21, (__int64 *)&v37, &v39);
  v24 = (__int64)v37;
  v35 = *(_DWORD *)(v20 + 48) + 1;
  v28 = v39;
LABEL_44:
  *(_DWORD *)(v20 + 48) = v35;
  if ( *v28 != -4096 )
    --*(_DWORD *)(v20 + 52);
  v30 = v28 + 1;
  *v28 = v24;
  v28[1] = 0;
  if ( v28 + 1 != (__int64 *)&v38 )
    goto LABEL_29;
  v31 = 0;
LABEL_48:
  v39 = (__int64 *)v31;
  LOBYTE(v40) = 1;
  if ( v38 )
    sub_B91220((__int64)&v38, (__int64)v38);
  return v39;
}

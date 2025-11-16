// Function: sub_19EC800
// Address: 0x19ec800
//
__int64 __fastcall sub_19EC800(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  __int64 result; // rax
  unsigned int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  char v28; // r8
  __int64 v29; // rax
  unsigned int v30; // esi
  int v31; // ecx
  int v32; // edx
  __int64 v33; // rdx
  char v34; // r8
  _QWORD *v35; // rax
  __int64 v36; // rdx
  unsigned int v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  unsigned int v39; // [rsp+24h] [rbp-9Ch]
  __int64 *v40; // [rsp+28h] [rbp-98h]
  int v41; // [rsp+28h] [rbp-98h]
  unsigned int v42; // [rsp+3Ch] [rbp-84h] BYREF
  __int64 v43[4]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v44; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h]
  __int64 v47; // [rsp+78h] [rbp-48h]

  v2 = a1 + 112;
  v3 = a2;
  ++*(_DWORD *)a1;
  v44 = (_QWORD *)a2;
  v4 = sub_19EC730(a1 + 112, (__int64 *)&v44);
  v39 = *(_DWORD *)a1;
  *(_DWORD *)(v4 + 8) = *(_DWORD *)a1;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v5 = *(__int64 **)(a2 - 8);
    v40 = &v5[3 * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v40 = (__int64 *)a2;
    v5 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  }
  if ( v40 != v5 )
  {
    while ( 1 )
    {
      v6 = *v5;
      if ( *(_BYTE *)(*v5 + 16) > 0x17u )
        break;
LABEL_11:
      v5 += 3;
      if ( v40 == v5 )
      {
        v3 = a2;
        goto LABEL_13;
      }
    }
    if ( (unsigned int)sub_19E5210(v2, *v5) )
    {
      v7 = *(_QWORD *)(a1 + 24);
      if ( v7 != *(_QWORD *)(a1 + 16) )
      {
LABEL_7:
        v8 = v7 + 8LL * *(unsigned int *)(a1 + 32);
        goto LABEL_8;
      }
    }
    else
    {
      sub_19EC800(a1, v6);
      v6 = *v5;
      v7 = *(_QWORD *)(a1 + 24);
      if ( v7 != *(_QWORD *)(a1 + 16) )
        goto LABEL_7;
    }
    v8 = v7 + 8LL * *(unsigned int *)(a1 + 36);
LABEL_8:
    v44 = (_QWORD *)v8;
    v45 = v8;
    sub_19E4730((__int64)&v44);
    v9 = *(_QWORD *)(a1 + 8);
    v46 = a1 + 8;
    v47 = v9;
    v10 = sub_15CC2D0(a1 + 8, v6);
    v11 = *(_QWORD *)(a1 + 24);
    if ( v11 == *(_QWORD *)(a1 + 16) )
      v12 = *(unsigned int *)(a1 + 36);
    else
      v12 = *(unsigned int *)(a1 + 32);
    v43[0] = (__int64)v10;
    v43[1] = v11 + 8 * v12;
    sub_19E4730((__int64)v43);
    if ( (_QWORD *)v43[0] == v44 )
    {
      LODWORD(v43[0]) = sub_19E5210(v2, *v5);
      v37 = v43[0];
      v16 = sub_19E5210(v2, a2);
      v17 = v43;
      v42 = v16;
      if ( v37 >= v16 )
        v17 = (__int64 *)&v42;
      v44 = (_QWORD *)a2;
      *(_DWORD *)(sub_19EC730(v2, (__int64 *)&v44) + 8) = *(_DWORD *)v17;
    }
    goto LABEL_11;
  }
LABEL_13:
  if ( v39 != (unsigned int)sub_19E5210(v2, v3) )
  {
    result = *(unsigned int *)(a1 + 152);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 156) )
    {
      sub_16CD150(a1 + 144, (const void *)(a1 + 160), 0, 8, v13, v14);
      result = *(unsigned int *)(a1 + 152);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * result) = v3;
    ++*(_DWORD *)(a1 + 152);
    return result;
  }
  v18 = *(unsigned int *)(a1 + 232);
  v19 = v18 + 1;
  v41 = *(_DWORD *)(a1 + 232);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 236) )
  {
    sub_19E6050(a1 + 224, v18 + 1);
    v18 = *(unsigned int *)(a1 + 232);
  }
  v20 = *(_QWORD *)(a1 + 224);
  v21 = v20 + 104 * v18;
  v22 = v20 + 104 * v19;
  if ( v21 != v22 )
  {
    do
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = 0;
        *(_QWORD *)(v21 + 8) = v21 + 40;
        *(_QWORD *)(v21 + 16) = v21 + 40;
        *(_DWORD *)(v21 + 24) = 8;
        *(_DWORD *)(v21 + 28) = 0;
        *(_DWORD *)(v21 + 32) = 0;
      }
      v21 += 104;
    }
    while ( v22 != v21 );
    v20 = *(_QWORD *)(a1 + 224);
  }
  *(_DWORD *)(a1 + 232) = v41 + 1;
  v23 = v20 + 104LL * (unsigned int)v19 - 104;
  sub_19E5420((__int64)&v44, v23, v3);
  sub_19E5420((__int64)&v44, a1 + 8, v3);
  v44 = (_QWORD *)v3;
  v38 = a1 + 1072;
  *(_DWORD *)(sub_19EC730(a1 + 1072, (__int64 *)&v44) + 8) = v41;
  result = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)result )
  {
    while ( 1 )
    {
      v43[0] = *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * result - 8);
      v34 = sub_154CC80(a1 + 112, v43, &v44);
      result = 0;
      if ( v34 )
        result = *((unsigned int *)v44 + 2);
      if ( v39 > (unsigned int)result )
        return result;
      v43[0] = *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8LL * *(unsigned int *)(a1 + 152) - 8);
      v35 = sub_1412190(v23, v43[0]);
      v36 = *(_QWORD *)(v23 + 16);
      v24 = v36 == *(_QWORD *)(v23 + 8) ? *(unsigned int *)(v23 + 28) : *(unsigned int *)(v23 + 24);
      v44 = v35;
      v45 = v36 + 8 * v24;
      sub_19E4730((__int64)&v44);
      v25 = sub_1412190(a1 + 8, v43[0]);
      v26 = *(_QWORD *)(a1 + 24);
      v27 = v26 == *(_QWORD *)(a1 + 16) ? *(unsigned int *)(a1 + 36) : *(unsigned int *)(a1 + 32);
      v44 = v25;
      v45 = v26 + 8 * v27;
      sub_19E4730((__int64)&v44);
      v28 = sub_154CC80(v38, v43, &v44);
      v29 = (__int64)v44;
      if ( !v28 )
        break;
LABEL_43:
      *(_DWORD *)(v29 + 8) = v41;
      result = (unsigned int)(*(_DWORD *)(a1 + 152) - 1);
      *(_DWORD *)(a1 + 152) = result;
      if ( !(_DWORD)result )
        return result;
    }
    v30 = *(_DWORD *)(a1 + 1096);
    v31 = *(_DWORD *)(a1 + 1088);
    ++*(_QWORD *)(a1 + 1072);
    v32 = v31 + 1;
    if ( 4 * (v31 + 1) >= 3 * v30 )
    {
      v30 *= 2;
    }
    else if ( v30 - *(_DWORD *)(a1 + 1092) - v32 > v30 >> 3 )
    {
LABEL_40:
      *(_DWORD *)(a1 + 1088) = v32;
      if ( *(_QWORD *)v29 != -8 )
        --*(_DWORD *)(a1 + 1092);
      v33 = v43[0];
      *(_DWORD *)(v29 + 8) = 0;
      *(_QWORD *)v29 = v33;
      goto LABEL_43;
    }
    sub_1542080(v38, v30);
    sub_154CC80(v38, v43, &v44);
    v29 = (__int64)v44;
    v32 = *(_DWORD *)(a1 + 1088) + 1;
    goto LABEL_40;
  }
  return result;
}

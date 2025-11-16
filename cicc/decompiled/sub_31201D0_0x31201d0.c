// Function: sub_31201D0
// Address: 0x31201d0
//
signed __int64 __fastcall sub_31201D0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  signed __int64 result; // rax
  unsigned __int64 v4; // r9
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // ecx
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // esi
  unsigned int v12; // ecx
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // edi
  unsigned int v17; // r11d
  unsigned __int64 v18; // r10
  int v19; // eax
  __int64 v20; // rax
  _DWORD *v21; // rsi
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // r15
  unsigned int v25; // edx
  int v26; // ecx
  int v27; // edx
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned int v30; // eax
  unsigned int v31; // eax
  int v32; // edx
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned int v36; // r11d
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rdx
  int v40; // esi
  __int64 v41; // rax
  _DWORD *v42; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 256 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v24 = a2;
LABEL_52:
    sub_31200F0(a1, (unsigned int *)v24, v24);
    do
    {
      v37 = *(_QWORD *)(v24 - 16);
      v24 -= 16LL;
      v38 = *(_QWORD *)(v24 + 8);
      *(_DWORD *)v24 = *(_DWORD *)a1;
      *(_DWORD *)(v24 + 4) = *(_DWORD *)(a1 + 4);
      *(_QWORD *)(v24 + 8) = *(_QWORD *)(a1 + 8);
      result = (signed __int64)sub_311D5D0(a1, 0, (__int64)(v24 - a1) >> 4, v37, v38);
    }
    while ( (__int64)(v24 - a1) > 16 );
    return result;
  }
  v7 = a1 + 16;
  v42 = (_DWORD *)(a1 + 32);
  while ( 2 )
  {
    v8 = *(_DWORD *)(a1 + 16);
    --v6;
    v9 = a1 + 16 * (result >> 5);
    v10 = *(_DWORD *)v9;
    if ( v8 >= *(_DWORD *)v9
      && (v8 != v10 || *(_DWORD *)(a1 + 20) >= *(_DWORD *)(v9 + 4))
      && (v8 > v10 || *(_DWORD *)(v9 + 4) < *(_DWORD *)(a1 + 20) || *(_QWORD *)(a1 + 24) >= *(_QWORD *)(v9 + 8)) )
    {
      v11 = *(_DWORD *)(v4 - 16);
      if ( v8 < v11 )
      {
        v17 = *(_DWORD *)(a1 + 20);
        v18 = *(_QWORD *)(a1 + 24);
LABEL_15:
        v19 = *(_DWORD *)(a1 + 4);
        v16 = *(_DWORD *)a1;
        *(_DWORD *)(a1 + 4) = v17;
        *(_DWORD *)a1 = v8;
        *(_DWORD *)(a1 + 20) = v19;
        v20 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 16) = v16;
        *(_QWORD *)(a1 + 8) = v18;
        *(_QWORD *)(a1 + 24) = v20;
        goto LABEL_16;
      }
      if ( v8 == v11 )
      {
        v17 = *(_DWORD *)(a1 + 20);
        if ( v17 < *(_DWORD *)(v4 - 12) )
        {
          v18 = *(_QWORD *)(a1 + 24);
          goto LABEL_15;
        }
        if ( v17 <= *(_DWORD *)(v4 - 12) )
        {
          v18 = *(_QWORD *)(a1 + 24);
          if ( v18 < *(_QWORD *)(v4 - 8) )
            goto LABEL_15;
        }
      }
      if ( v10 >= v11
        && (v10 != v11 || *(_DWORD *)(v9 + 4) >= *(_DWORD *)(v4 - 12))
        && (v10 > v11 || *(_DWORD *)(v4 - 12) < *(_DWORD *)(v9 + 4) || *(_QWORD *)(v9 + 8) >= *(_QWORD *)(v4 - 8)) )
      {
        goto LABEL_10;
      }
LABEL_32:
      v31 = *(_DWORD *)a1;
      *(_DWORD *)a1 = v11;
      v32 = *(_DWORD *)(v4 - 12);
      *(_DWORD *)(v4 - 16) = v31;
      v33 = *(_DWORD *)(a1 + 4);
      *(_DWORD *)(a1 + 4) = v32;
      v34 = *(_QWORD *)(v4 - 8);
      *(_DWORD *)(v4 - 12) = v33;
      v35 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v34;
      *(_QWORD *)(v4 - 8) = v35;
      v16 = *(_DWORD *)(a1 + 16);
      v8 = *(_DWORD *)a1;
      goto LABEL_16;
    }
    v11 = *(_DWORD *)(v4 - 16);
    if ( v10 < v11
      || v10 == v11 && *(_DWORD *)(v9 + 4) < *(_DWORD *)(v4 - 12)
      || v10 <= v11 && *(_DWORD *)(v4 - 12) >= *(_DWORD *)(v9 + 4) && *(_QWORD *)(v9 + 8) < *(_QWORD *)(v4 - 8) )
    {
LABEL_10:
      v12 = *(_DWORD *)a1;
      *(_DWORD *)a1 = v10;
      *(_DWORD *)v9 = v12;
      v13 = *(_DWORD *)(a1 + 4);
      *(_DWORD *)(a1 + 4) = *(_DWORD *)(v9 + 4);
      v14 = *(_QWORD *)(v9 + 8);
      *(_DWORD *)(v9 + 4) = v13;
      v15 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = v14;
      *(_QWORD *)(v9 + 8) = v15;
      v16 = *(_DWORD *)(a1 + 16);
      v8 = *(_DWORD *)a1;
      goto LABEL_16;
    }
    if ( v8 < v11 )
      goto LABEL_32;
    v30 = *(_DWORD *)(a1 + 20);
    if ( v8 == v11 && *(_DWORD *)(v4 - 12) > v30 )
      goto LABEL_32;
    v39 = *(_QWORD *)(a1 + 24);
    if ( v8 <= v11 && *(_DWORD *)(v4 - 12) >= v30 && *(_QWORD *)(v4 - 8) > v39 )
      goto LABEL_32;
    v16 = *(_DWORD *)a1;
    v40 = *(_DWORD *)(a1 + 4);
    *(_DWORD *)(a1 + 4) = v30;
    v41 = *(_QWORD *)(a1 + 8);
    *(_DWORD *)a1 = v8;
    *(_DWORD *)(a1 + 16) = v16;
    *(_DWORD *)(a1 + 20) = v40;
    *(_QWORD *)(a1 + 8) = v39;
    *(_QWORD *)(a1 + 24) = v41;
LABEL_16:
    v21 = v42;
    v22 = v7;
    v23 = v4;
    while ( 1 )
    {
      v24 = v22;
      if ( v8 <= v16
        && (v8 != v16 || *(v21 - 3) >= *(_DWORD *)(a1 + 4))
        && (v8 < v16 || *(_DWORD *)(a1 + 4) < *(v21 - 3) || *((_QWORD *)v21 - 1) >= *(_QWORD *)(a1 + 8)) )
      {
        break;
      }
LABEL_19:
      v16 = *v21;
      v22 += 16LL;
      v21 += 4;
    }
    while ( 1 )
    {
      do
      {
        v25 = *(_DWORD *)(v23 - 16);
        v23 -= 16LL;
      }
      while ( v25 > v8 );
      if ( v25 != v8 )
      {
LABEL_26:
        if ( v22 >= v23 )
          goto LABEL_37;
LABEL_27:
        *(v21 - 4) = v25;
        v26 = *(_DWORD *)(v23 + 4);
        *(_DWORD *)v23 = v16;
        v27 = *(v21 - 3);
        *(v21 - 3) = v26;
        v28 = *(_QWORD *)(v23 + 8);
        *(_DWORD *)(v23 + 4) = v27;
        v29 = *((_QWORD *)v21 - 1);
        *((_QWORD *)v21 - 1) = v28;
        *(_QWORD *)(v23 + 8) = v29;
        v8 = *(_DWORD *)a1;
        goto LABEL_19;
      }
      v36 = *(_DWORD *)(v23 + 4);
      if ( *(_DWORD *)(a1 + 4) >= v36 )
      {
        if ( *(_DWORD *)(a1 + 4) > v36 )
          goto LABEL_26;
        if ( *(_QWORD *)(a1 + 8) >= *(_QWORD *)(v23 + 8) )
          break;
      }
    }
    if ( v22 < v23 )
      goto LABEL_27;
LABEL_37:
    sub_31201D0(v22, v4, v6);
    result = v22 - a1;
    if ( (__int64)(v22 - a1) > 256 )
    {
      if ( v6 )
      {
        v4 = v22;
        continue;
      }
      goto LABEL_52;
    }
    return result;
  }
}

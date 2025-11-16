// Function: sub_1873100
// Address: 0x1873100
//
__int64 __fastcall sub_1873100(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // r8
  __int64 v8; // r12
  unsigned int v9; // ecx
  unsigned int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // esi
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  int v21; // ecx
  int v22; // eax
  int v23; // ecx
  int v24; // edx
  __int64 v25; // rbx
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v28; // rcx
  unsigned int v29; // r8d
  unsigned __int64 v30; // r13
  __int64 v31; // rcx
  unsigned int v32; // r8d
  __int64 v33; // rbx

  result = a2 - a1;
  if ( (__int64)(a2 - a1) <= 256 )
    return result;
  v4 = a3;
  v6 = a2;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 16;
  while ( 2 )
  {
    v9 = *(_DWORD *)(a1 + 24);
    v10 = *(_DWORD *)(v7 - 8);
    --v4;
    v11 = *(_QWORD *)a1;
    v12 = a1 + 16 * ((__int64)(((__int64)(v7 - a1) >> 4) + ((v7 - a1) >> 63)) >> 1);
    v13 = *(_DWORD *)(v12 + 8);
    if ( v9 >= v13 )
    {
      if ( v9 < v10 )
        goto LABEL_7;
      if ( v13 < v10 )
      {
LABEL_18:
        *(_QWORD *)a1 = *(_QWORD *)(v7 - 16);
        v22 = *(_DWORD *)(v7 - 8);
        *(_QWORD *)(v7 - 16) = v11;
        v16 = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v22;
        *(_DWORD *)(v7 - 8) = v16;
        v15 = *(_DWORD *)(a1 + 24);
        v9 = *(_DWORD *)(a1 + 8);
        goto LABEL_8;
      }
LABEL_23:
      *(_QWORD *)a1 = *(_QWORD *)v12;
      v23 = *(_DWORD *)(v12 + 8);
      *(_QWORD *)v12 = v11;
      v24 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(a1 + 8) = v23;
      *(_DWORD *)(v12 + 8) = v24;
      v15 = *(_DWORD *)(a1 + 24);
      v9 = *(_DWORD *)(a1 + 8);
      v16 = *(_DWORD *)(v7 - 8);
      goto LABEL_8;
    }
    if ( v13 < v10 )
      goto LABEL_23;
    if ( v9 < v10 )
      goto LABEL_18;
LABEL_7:
    v14 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v11;
    *(_QWORD *)a1 = v14;
    v15 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v9;
    *(_DWORD *)(a1 + 24) = v15;
    v16 = *(_DWORD *)(v7 - 8);
LABEL_8:
    v17 = v8;
    v18 = v7;
    while ( 1 )
    {
      v6 = v17;
      if ( v9 > v15 )
        goto LABEL_15;
      if ( v16 <= v9 )
      {
        v18 -= 16LL;
      }
      else
      {
        v19 = v18 - 32;
        do
        {
          v18 = v19;
          v19 -= 16LL;
        }
        while ( v9 < *(_DWORD *)(v19 + 24) );
      }
      if ( v17 >= v18 )
        break;
      v20 = *(_QWORD *)v17;
      *(_QWORD *)v17 = *(_QWORD *)v18;
      v21 = *(_DWORD *)(v18 + 8);
      *(_QWORD *)v18 = v20;
      LODWORD(v20) = *(_DWORD *)(v17 + 8);
      *(_DWORD *)(v17 + 8) = v21;
      v16 = *(_DWORD *)(v18 - 8);
      *(_DWORD *)(v18 + 8) = v20;
      v9 = *(_DWORD *)(a1 + 8);
LABEL_15:
      v15 = *(_DWORD *)(v17 + 24);
      v17 += 16LL;
    }
    sub_1873100(v17, v7, v4);
    result = v17 - a1;
    if ( (__int64)(v17 - a1) > 256 )
    {
      if ( v4 )
      {
        v7 = v17;
        continue;
      }
LABEL_24:
      v25 = result >> 4;
      v26 = ((result >> 4) - 2) >> 1;
      v27 = a1 + 16 * v26;
      while ( 1 )
      {
        v28 = *(_QWORD *)v27;
        v29 = *(_DWORD *)(v27 + 8);
        v27 -= 16;
        sub_1872830(a1, v26, v25, v28, v29);
        if ( !v26 )
          break;
        --v26;
      }
      v30 = v6 - 16;
      do
      {
        v31 = *(_QWORD *)v30;
        v32 = *(_DWORD *)(v30 + 8);
        v33 = v30 - a1;
        v30 -= 16LL;
        *(_QWORD *)(v30 + 16) = *(_QWORD *)a1;
        *(_DWORD *)(v30 + 24) = *(_DWORD *)(a1 + 8);
        result = sub_1872830(a1, 0, v33 >> 4, v31, v32);
      }
      while ( v33 > 16 );
    }
    return result;
  }
}

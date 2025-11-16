// Function: sub_18DB560
// Address: 0x18db560
//
__int64 __fastcall sub_18DB560(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r14
  __int64 v7; // rsi
  __int64 *v8; // r13
  int v9; // edx
  __int64 *v10; // rax
  __int64 *v11; // r14
  __int64 v12; // rsi
  __int64 *v13; // r13
  __int64 *v15; // r9
  __int64 *v16; // r8
  __int64 *v17; // rax
  __int64 *v18; // rdi
  unsigned int v19; // r10d
  __int64 *v20; // rax
  __int64 *v21; // rcx
  __int64 *v22; // r9
  __int64 *v23; // r8
  int v24; // edx
  __int64 *v25; // rax
  __int64 *v26; // rdi
  unsigned int v27; // r10d
  __int64 *v28; // rax
  __int64 *v29; // rcx

  v2 = a2;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
    *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)a1 &= *(_BYTE *)a2;
  *(_BYTE *)(a1 + 1) &= *(_BYTE *)(a2 + 1);
  *(_BYTE *)(a1 + 128) |= *(_BYTE *)(a2 + 128);
  v4 = *(__int64 **)(a2 + 32);
  if ( v4 == *(__int64 **)(a2 + 24) )
    v5 = *(unsigned int *)(a2 + 44);
  else
    v5 = *(unsigned int *)(a2 + 40);
  v6 = &v4[v5];
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      v7 = *v4;
      v8 = v4;
      if ( (unsigned __int64)*v4 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == ++v4 )
        goto LABEL_8;
    }
    if ( v4 != v6 )
    {
      v15 = *(__int64 **)(a1 + 32);
      v16 = *(__int64 **)(a1 + 24);
      if ( v16 == v15 )
        goto LABEL_23;
LABEL_16:
      sub_16CCBA0(a1 + 16, v7);
      v15 = *(__int64 **)(a1 + 32);
      v16 = *(__int64 **)(a1 + 24);
LABEL_17:
      while ( 1 )
      {
        v17 = v8 + 1;
        if ( v8 + 1 == v6 )
          break;
        v7 = *v17;
        for ( ++v8; (unsigned __int64)*v17 >= 0xFFFFFFFFFFFFFFFELL; v8 = v17 )
        {
          if ( v6 == ++v17 )
            goto LABEL_8;
          v7 = *v17;
        }
        if ( v8 == v6 )
          break;
        if ( v16 != v15 )
          goto LABEL_16;
LABEL_23:
        v18 = &v16[*(unsigned int *)(a1 + 44)];
        v19 = *(_DWORD *)(a1 + 44);
        if ( v16 == v18 )
        {
LABEL_50:
          if ( v19 >= *(_DWORD *)(a1 + 40) )
            goto LABEL_16;
          *(_DWORD *)(a1 + 44) = v19 + 1;
          *v18 = v7;
          v16 = *(__int64 **)(a1 + 24);
          ++*(_QWORD *)(a1 + 16);
          v15 = *(__int64 **)(a1 + 32);
        }
        else
        {
          v20 = v16;
          v21 = 0;
          while ( *v20 != v7 )
          {
            if ( *v20 == -2 )
              v21 = v20;
            if ( v18 == ++v20 )
            {
              if ( !v21 )
                goto LABEL_50;
              *v21 = v7;
              v15 = *(__int64 **)(a1 + 32);
              --*(_DWORD *)(a1 + 48);
              v16 = *(__int64 **)(a1 + 24);
              ++*(_QWORD *)(a1 + 16);
              goto LABEL_17;
            }
          }
        }
      }
    }
  }
LABEL_8:
  v9 = *(_DWORD *)(v2 + 100);
  v10 = *(__int64 **)(v2 + 88);
  if ( v10 == *(__int64 **)(v2 + 80) )
    v11 = &v10[v9];
  else
    v11 = &v10[*(unsigned int *)(v2 + 96)];
  LOBYTE(v2) = *(_DWORD *)(a1 + 100) - *(_DWORD *)(a1 + 104) != v9 - *(_DWORD *)(v2 + 104);
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v12 = *v10;
      v13 = v10;
      if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v11 == ++v10 )
        return (unsigned int)v2;
    }
    if ( v10 != v11 )
    {
      v22 = *(__int64 **)(a1 + 88);
      v23 = *(__int64 **)(a1 + 80);
      if ( v23 == v22 )
        goto LABEL_40;
LABEL_33:
      sub_16CCBA0(a1 + 72, v12);
      v22 = *(__int64 **)(a1 + 88);
      v23 = *(__int64 **)(a1 + 80);
      LODWORD(v2) = v24 | v2;
LABEL_34:
      while ( 1 )
      {
        v25 = v13 + 1;
        if ( v13 + 1 == v11 )
          break;
        v12 = *v25;
        for ( ++v13; (unsigned __int64)*v25 >= 0xFFFFFFFFFFFFFFFELL; v13 = v25 )
        {
          if ( v11 == ++v25 )
            return (unsigned int)v2;
          v12 = *v25;
        }
        if ( v13 == v11 )
          return (unsigned int)v2;
        if ( v23 != v22 )
          goto LABEL_33;
LABEL_40:
        v26 = &v23[*(unsigned int *)(a1 + 100)];
        v27 = *(_DWORD *)(a1 + 100);
        if ( v23 == v26 )
        {
LABEL_48:
          if ( v27 >= *(_DWORD *)(a1 + 96) )
            goto LABEL_33;
          LODWORD(v2) = 1;
          *(_DWORD *)(a1 + 100) = v27 + 1;
          *v26 = v12;
          v23 = *(__int64 **)(a1 + 80);
          ++*(_QWORD *)(a1 + 72);
          v22 = *(__int64 **)(a1 + 88);
        }
        else
        {
          v28 = v23;
          v29 = 0;
          while ( *v28 != v12 )
          {
            if ( *v28 == -2 )
              v29 = v28;
            if ( v26 == ++v28 )
            {
              if ( !v29 )
                goto LABEL_48;
              *v29 = v12;
              LODWORD(v2) = 1;
              v22 = *(__int64 **)(a1 + 88);
              --*(_DWORD *)(a1 + 104);
              v23 = *(__int64 **)(a1 + 80);
              ++*(_QWORD *)(a1 + 72);
              goto LABEL_34;
            }
          }
        }
      }
    }
  }
  return (unsigned int)v2;
}

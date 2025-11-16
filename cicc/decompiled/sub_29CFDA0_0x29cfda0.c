// Function: sub_29CFDA0
// Address: 0x29cfda0
//
__int64 __fastcall sub_29CFDA0(_QWORD *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v8; // edx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int8 *v17; // r15
  unsigned __int8 *v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  _BYTE *v21; // rdx
  _BYTE *v22; // r13
  __int64 v23; // rcx
  __int64 v24; // rdi
  unsigned int v25; // esi
  __int64 *v26; // rcx
  __int64 v27; // rcx
  int v28; // ecx
  int v29; // r10d

  result = 0;
  if ( *(_QWORD *)(a3 + 24) != *((_QWORD *)a2 + 10) )
    return result;
  v8 = *a2;
  if ( v8 == 40 )
  {
    v10 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v10 = -32;
    if ( v8 != 85 )
    {
      v10 = -96;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v11 = sub_BD2BC0((__int64)a2);
    v13 = v11 + v12;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v13 >> 4) )
        goto LABEL_33;
    }
    else if ( (unsigned int)((v13 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v14 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v15 = sub_BD2BC0((__int64)a2);
        v10 -= 32LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
        goto LABEL_10;
      }
LABEL_33:
      BUG();
    }
  }
LABEL_10:
  v17 = &a2[v10];
  v18 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v18 != v17 )
  {
    v19 = *(unsigned int *)(a4 + 8);
    while ( 1 )
    {
      v21 = *(_BYTE **)v18;
      v22 = *(_BYTE **)v18;
      if ( **(_BYTE **)v18 <= 0x15u )
        goto LABEL_12;
      v23 = a1[6];
      if ( v23 == a1[7] )
      {
        v27 = *(_QWORD *)(a1[9] - 8LL);
        a5 = *(unsigned int *)(v27 + 504);
        v24 = *(_QWORD *)(v27 + 488);
        if ( !(_DWORD)a5 )
          goto LABEL_24;
      }
      else
      {
        a5 = *(unsigned int *)(v23 - 8);
        v24 = *(_QWORD *)(v23 - 24);
        if ( !(_DWORD)a5 )
          goto LABEL_24;
      }
      a5 = (unsigned int)(a5 - 1);
      v25 = a5 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v26 = (__int64 *)(v24 + 16LL * v25);
      a6 = *v26;
      if ( v21 != (_BYTE *)*v26 )
        break;
LABEL_18:
      v22 = (_BYTE *)v26[1];
      v20 = v19 + 1;
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
LABEL_19:
        sub_C8D5F0(a4, (const void *)(a4 + 16), v20, 8u, a5, a6);
        v19 = *(unsigned int *)(a4 + 8);
      }
LABEL_13:
      v18 += 32;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v19) = v22;
      v19 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
      *(_DWORD *)(a4 + 8) = v19;
      if ( v17 == v18 )
        return 1;
    }
    v28 = 1;
    while ( a6 != -4096 )
    {
      v29 = v28 + 1;
      v25 = a5 & (v28 + v25);
      v26 = (__int64 *)(v24 + 16LL * v25);
      a6 = *v26;
      if ( v21 == (_BYTE *)*v26 )
        goto LABEL_18;
      v28 = v29;
    }
LABEL_24:
    v22 = 0;
LABEL_12:
    v20 = v19 + 1;
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      goto LABEL_19;
    goto LABEL_13;
  }
  return 1;
}

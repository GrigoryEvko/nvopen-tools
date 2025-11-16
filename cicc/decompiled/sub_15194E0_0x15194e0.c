// Function: sub_15194E0
// Address: 0x15194e0
//
__int64 __fastcall sub_15194E0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  char v5; // al
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  bool v8; // cf
  __int64 v9; // rcx
  __int64 *v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // edi
  unsigned int v16; // eax
  __int64 result; // rax
  __int64 *v18; // rdi
  unsigned __int64 v19; // r14
  int v20; // r15d
  __int64 v21; // r12
  __int64 v22; // r14
  _QWORD *v23; // rax
  unsigned int v24; // r9d
  unsigned int v25; // [rsp+Ch] [rbp-64h] BYREF
  char v26[96]; // [rsp+10h] [rbp-60h] BYREF

  v3 = a2;
  v5 = *(_BYTE *)a2;
  v25 = a3;
  if ( (unsigned __int8)(v5 - 4) > 0x1Eu || *(_BYTE *)(a2 + 1) != 2 && !*(_DWORD *)(a2 + 12) )
  {
    v6 = *(unsigned int *)(a1 + 8);
    v7 = v25;
    v8 = v25 < (unsigned int)v6;
    if ( v25 != (_DWORD)v6 )
      goto LABEL_5;
LABEL_19:
    if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 12) )
    {
      sub_1516630(a1, 0);
      LODWORD(v6) = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)v6;
    v18 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)v6);
    if ( v18 )
    {
      *v18 = v3;
      result = sub_1623A60(v18, v3, 2);
      LODWORD(v6) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v6 + 1;
    return result;
  }
  a2 = a1 + 56;
  sub_1517B60((__int64)v26, a1 + 56, (int *)&v25);
  v6 = *(unsigned int *)(a1 + 8);
  v7 = v25;
  v8 = v25 < (unsigned int)v6;
  if ( v25 == (_DWORD)v6 )
    goto LABEL_19;
LABEL_5:
  if ( !v8 )
  {
    v19 = (unsigned int)(v7 + 1);
    v20 = v7 + 1;
    if ( v19 < v6 )
    {
      v9 = *(_QWORD *)a1;
      v21 = *(_QWORD *)a1 + 8 * v6;
      v22 = *(_QWORD *)a1 + 8 * v19;
      if ( v21 != v22 )
      {
        do
        {
          a2 = *(_QWORD *)(v21 - 8);
          v21 -= 8;
          if ( a2 )
            sub_161E7C0(v21);
        }
        while ( v22 != v21 );
        v7 = v25;
        v9 = *(_QWORD *)a1;
      }
      *(_DWORD *)(a1 + 8) = v20;
      goto LABEL_7;
    }
    if ( v19 > v6 )
    {
      if ( v19 > *(unsigned int *)(a1 + 12) )
      {
        a2 = (unsigned int)(v7 + 1);
        sub_1516630(a1, v19);
        v6 = *(unsigned int *)(a1 + 8);
      }
      v9 = *(_QWORD *)a1;
      v23 = (_QWORD *)(*(_QWORD *)a1 + 8 * v6);
      v6 = *(_QWORD *)a1 + 8 * v19;
      if ( v23 != (_QWORD *)v6 )
      {
        do
        {
          if ( v23 )
            *v23 = 0;
          ++v23;
        }
        while ( (_QWORD *)v6 != v23 );
        v9 = *(_QWORD *)a1;
      }
      *(_DWORD *)(a1 + 8) = v19;
      v7 = v25;
      goto LABEL_7;
    }
  }
  v9 = *(_QWORD *)a1;
LABEL_7:
  v10 = (__int64 *)(v9 + 8 * v7);
  v11 = *v10;
  if ( *v10 )
  {
    v12 = *(_QWORD *)(v11 + 16);
    if ( (v12 & 4) != 0 )
    {
      a2 = v3;
      sub_16302D0(v12 & 0xFFFFFFFFFFFFFFF8LL, v3);
      if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
        goto LABEL_10;
    }
    else if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    {
LABEL_10:
      v13 = *(unsigned int *)(a1 + 48);
      v14 = *(_QWORD *)(a1 + 40);
      if ( !(_DWORD)v13 )
        return sub_16307F0(v11, a2, v6, v13, v14);
      v13 = (unsigned int)(v13 - 1);
LABEL_12:
      v6 = (unsigned int)v13 & (37 * v25);
      a2 = v14 + 4 * v6;
      v15 = *(_DWORD *)a2;
      if ( v25 == *(_DWORD *)a2 )
      {
LABEL_13:
        *(_DWORD *)a2 = -2;
        v16 = *(_DWORD *)(a1 + 32);
        ++*(_DWORD *)(a1 + 36);
        v6 = 2 * (v16 >> 1) - 2;
        *(_DWORD *)(a1 + 32) = v6 | v16 & 1;
      }
      else
      {
        a2 = 1;
        while ( v15 != -1 )
        {
          v24 = a2 + 1;
          v6 = (unsigned int)v13 & ((_DWORD)a2 + (_DWORD)v6);
          a2 = v14 + 4LL * (unsigned int)v6;
          v15 = *(_DWORD *)a2;
          if ( v25 == *(_DWORD *)a2 )
            goto LABEL_13;
          a2 = v24;
        }
      }
      return sub_16307F0(v11, a2, v6, v13, v14);
    }
    v14 = a1 + 40;
    v13 = 0;
    goto LABEL_12;
  }
  *v10 = v3;
  return sub_1623A60(v10, v3, 2);
}

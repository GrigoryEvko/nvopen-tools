// Function: sub_2F462B0
// Address: 0x2f462b0
//
__int64 __fastcall sub_2F462B0(_QWORD *a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r13
  bool v13; // al
  bool v14; // zf
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  __int64 v21; // [rsp+10h] [rbp-40h]

  v4 = a1[92];
  v21 = 1LL << a2;
  v5 = a1[48];
  if ( (*(_QWORD *)(v4 + 8LL * ((a2 >> 6) & 0x1FFFFFF)) & (1LL << a2)) != 0 )
  {
    LOBYTE(v2) = *(_DWORD *)(v5 + 120) != 0;
  }
  else
  {
    LOBYTE(v7) = sub_2E322C0(v5, v5);
    v8 = a2;
    v2 = v7;
    if ( (_BYTE)v7 )
    {
      v9 = a1[1];
      if ( (a2 & 0x80000000) != 0 )
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
      else
        v10 = *(_QWORD *)(*(_QWORD *)(v9 + 304) + 8LL * a2);
      if ( !v10 )
        goto LABEL_43;
      if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
      {
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 || (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
          goto LABEL_43;
      }
      v11 = *(_QWORD *)(v10 + 16);
      v12 = 0;
LABEL_10:
      if ( a1[48] != *(_QWORD *)(v11 + 24) )
      {
LABEL_43:
        *(_QWORD *)(a1[92] + 8LL * ((a2 >> 6) & 0x1FFFFFF)) |= v21;
        return v2;
      }
      if ( v12 )
      {
        v20 = v8;
        v13 = sub_2F46050((__int64)(a1 + 155), v11, v12);
        v8 = v20;
        v14 = !v13;
        v15 = *(_QWORD *)(v10 + 16);
        if ( !v14 )
          v12 = v11;
      }
      else
      {
        v12 = v11;
        v15 = v11;
      }
      while ( 1 )
      {
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 || (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
          break;
        v11 = *(_QWORD *)(v10 + 16);
        if ( v15 != v11 )
          goto LABEL_10;
      }
    }
    else
    {
      v12 = 0;
    }
    v16 = a1[1];
    if ( (int)v8 < 0 )
      v17 = *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16 * (v8 & 0x7FFFFFFF) + 8);
    else
      v17 = *(_QWORD *)(*(_QWORD *)(v16 + 304) + 8 * v8);
    while ( 1 )
    {
      if ( !v17 )
        return 0;
      if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
        break;
      v17 = *(_QWORD *)(v17 + 32);
    }
    v18 = *(_QWORD *)(v17 + 16);
    v2 = 8;
LABEL_28:
    if ( a1[48] == *(_QWORD *)(v18 + 24) && (--v2, v2) )
    {
      v19 = v18;
      if ( !v12 )
        goto LABEL_34;
      if ( v12 != v18 && sub_2F46050((__int64)(a1 + 155), v12, v18) )
      {
        v19 = *(_QWORD *)(v17 + 16);
LABEL_34:
        while ( 1 )
        {
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 )
            return 0;
          if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
          {
            v18 = *(_QWORD *)(v17 + 16);
            if ( v18 != v19 )
              goto LABEL_28;
          }
        }
      }
      v2 = 1;
      *(_QWORD *)(a1[92] + 8LL * ((a2 >> 6) & 0x1FFFFFF)) |= v21;
    }
    else
    {
      *(_QWORD *)(a1[92] + 8LL * ((a2 >> 6) & 0x1FFFFFF)) |= v21;
      LOBYTE(v2) = *(_DWORD *)(a1[48] + 120LL) != 0;
    }
  }
  return v2;
}

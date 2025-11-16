// Function: sub_27C1230
// Address: 0x27c1230
//
__int64 __fastcall sub_27C1230(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // r13d
  unsigned __int8 v4; // r15
  __int64 *v7; // rbx
  char v8; // al
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // r13
  __int64 *v17; // rax

  v4 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    LOBYTE(v3) = (unsigned __int8)(v4 - 12) > 1u;
    return v3;
  }
  LOBYTE(v3) = a3 == 6 || v4 <= 0x1Cu;
  if ( (_BYTE)v3 )
    return 0;
  v7 = (__int64 *)a1;
  v8 = sub_B46420(a1);
  LOBYTE(v11) = v4 == 34;
  LOBYTE(v12) = v4 == 34 || v4 == 85;
  if ( !(_BYTE)v12 && !v8 )
  {
    v13 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v14 = *(__int64 **)(a1 - 8);
      v7 = &v14[v13];
    }
    else
    {
      v14 = (__int64 *)(a1 - v13 * 8);
    }
    v15 = a3 + 1;
    if ( v14 == v7 )
      return 1;
    while ( 1 )
    {
      v16 = *v14;
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_19;
      v17 = *(__int64 **)(a2 + 8);
      v12 = *(unsigned int *)(a2 + 20);
      v11 = &v17[v12];
      if ( v17 != v11 )
      {
        while ( v16 != *v17 )
        {
          if ( v11 == ++v17 )
            goto LABEL_22;
        }
        goto LABEL_16;
      }
LABEL_22:
      if ( (unsigned int)v12 < *(_DWORD *)(a2 + 16) )
      {
        *(_DWORD *)(a2 + 20) = v12 + 1;
        *v11 = v16;
        ++*(_QWORD *)a2;
      }
      else
      {
LABEL_19:
        sub_C8CC70(a2, *v14, (__int64)v11, v12, v9, v10);
        if ( !(_BYTE)v11 )
          goto LABEL_16;
      }
      v3 = sub_27C1230(v16, a2, v15);
      if ( !(_BYTE)v3 )
        return v3;
LABEL_16:
      v14 += 4;
      if ( v7 == v14 )
        return 1;
    }
  }
  return v3;
}

// Function: sub_DE0740
// Address: 0xde0740
//
__int64 *__fastcall sub_DE0740(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rbx
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 v6; // r8
  char v7; // al
  __int64 *v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 *v13; // r13
  signed __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)a2;
  v3 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v4 = *(__int64 **)(a2 - 8);
    v5 = &v4[v3];
  }
  else
  {
    v4 = (__int64 *)(a2 - v3 * 8);
    v5 = (__int64 *)a2;
  }
  v6 = 0;
  if ( v4 != v5 )
  {
    do
    {
      if ( (unsigned __int8)(*(_BYTE *)*v4 - 42) > 0x11u )
        return 0;
      if ( v6 )
      {
        v16 = v6;
        v7 = sub_B46130(v6, *v4, 0);
        v6 = v16;
        if ( !v7 )
          return 0;
      }
      else
      {
        v6 = *v4;
      }
      v4 += 4;
    }
    while ( v5 != v4 );
    v8 = sub_DD8400(a1, v6);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v9 = *(_QWORD *)(a2 - 8);
      v10 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v2 = (__int64 *)(v9 + v10);
    }
    else
    {
      v10 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v9 = a2 - v10;
    }
    v11 = (__int64 *)(v9 + 32);
    v12 = (v10 - 32) >> 7;
    if ( v12 > 0 )
    {
      v13 = (__int64 *)(v9 + (v12 << 7) + 32);
      while ( v8 == sub_DD8400(a1, *v11) )
      {
        if ( v8 != sub_DD8400(a1, v11[4]) )
        {
          v11 += 4;
          break;
        }
        if ( v8 != sub_DD8400(a1, v11[8]) )
        {
          v11 += 8;
          break;
        }
        if ( v8 != sub_DD8400(a1, v11[12]) )
        {
          v11 += 12;
          break;
        }
        v11 += 16;
        if ( v11 == v13 )
          goto LABEL_24;
      }
LABEL_17:
      if ( v2 == v11 )
        return v8;
      return 0;
    }
    v13 = (__int64 *)(v9 + 32);
LABEL_24:
    v15 = (char *)v2 - (char *)v13;
    if ( (char *)v2 - (char *)v13 != 64 )
    {
      if ( v15 != 96 )
      {
        if ( v15 != 32 )
          return v8;
        goto LABEL_27;
      }
      v11 = v13;
      if ( v8 != sub_DD8400(a1, *v13) )
        goto LABEL_17;
      v13 += 4;
    }
    v11 = v13;
    if ( v8 != sub_DD8400(a1, *v13) )
      goto LABEL_17;
    v13 += 4;
LABEL_27:
    if ( v8 == sub_DD8400(a1, *v13) )
      return v8;
    v11 = v13;
    goto LABEL_17;
  }
  return 0;
}

// Function: sub_9BA900
// Address: 0x9ba900
//
__int64 __fastcall sub_9BA900(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 *v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rax
  int v7; // edi
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rcx
  _BYTE *i; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE *v17; // [rsp+10h] [rbp-60h] BYREF
  __int64 v18; // [rsp+18h] [rbp-58h]
  _BYTE v19[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = 0;
  v4 = (__int64 *)v19;
  v18 = 0x400000000LL;
  v5 = *(_DWORD *)(a1 + 32);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_DWORD *)(a1 + 24);
  v17 = v19;
  if ( v7 )
  {
    v9 = v6 + 16LL * v5;
    if ( v6 != v9 )
    {
      while ( 1 )
      {
        v10 = v6;
        if ( (unsigned int)(*(_DWORD *)v6 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v6 += 16;
        if ( v9 == v6 )
          goto LABEL_8;
      }
      if ( v9 == v6 )
      {
LABEL_8:
        v2 = 0;
        v4 = (__int64 *)v19;
        goto LABEL_2;
      }
      v11 = *(_QWORD *)(v6 + 8);
      v12 = 0;
      for ( i = v19; ; i = v17 )
      {
        *(_QWORD *)&i[8 * v12] = v11;
        LODWORD(v12) = v18 + 1;
        v14 = v10 + 16;
        LODWORD(v18) = v18 + 1;
        if ( v9 == v10 + 16 )
        {
LABEL_14:
          v4 = (__int64 *)v17;
          v2 = (unsigned int)v12;
          goto LABEL_2;
        }
        while ( 1 )
        {
          v10 = v14;
          if ( (unsigned int)(*(_DWORD *)v14 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          v14 += 16;
          if ( v9 == v14 )
            goto LABEL_14;
        }
        v2 = (unsigned int)v12;
        v12 = (unsigned int)v12;
        if ( v9 == v14 )
          break;
        v15 = (unsigned int)v12 + 1LL;
        v11 = *(_QWORD *)(v14 + 8);
        if ( v15 > HIDWORD(v18) )
        {
          v16 = *(_QWORD *)(v14 + 8);
          sub_C8D5F0(&v17, v19, v15, 8);
          v12 = (unsigned int)v18;
          v11 = v16;
        }
      }
      v4 = (__int64 *)v17;
    }
  }
LABEL_2:
  result = sub_9B8FE0(a2, v4, v2);
  if ( v17 != v19 )
    return _libc_free(v17, v4);
  return result;
}

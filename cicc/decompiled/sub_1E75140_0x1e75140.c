// Function: sub_1E75140
// Address: 0x1e75140
//
__int64 __fastcall sub_1E75140(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  int v7; // r9d
  __int64 i; // rcx
  unsigned __int8 v9; // dl
  unsigned int v10; // r11d
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 v13; // rdx
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // eax
  _DWORD v17[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v18; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)(v2 + 32);
  result = 5LL * *(unsigned int *)(v2 + 40);
  v5 = v3 + 40LL * *(unsigned int *)(v2 + 40);
  if ( v5 != v3 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v3 )
        goto LABEL_8;
      result = *(unsigned __int8 *)(v3 + 4);
      if ( (result & 1) != 0 || (result & 2) != 0 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
        break;
      v7 = *(_DWORD *)(v3 + 8);
      if ( !*(_BYTE *)(a1 + 914) )
        goto LABEL_21;
      if ( v7 < 0 )
      {
        result = *(_QWORD *)(v2 + 32);
        for ( i = result + 40LL * *(unsigned int *)(v2 + 40); i != result; result += 40 )
        {
          if ( !*(_BYTE *)result )
          {
            v9 = *(_BYTE *)(result + 3);
            if ( (v9 & 0x10) != 0 && v7 == *(_DWORD *)(result + 8) && (((v9 & 0x10) != 0) & (v9 >> 6)) == 0 )
              goto LABEL_8;
          }
        }
LABEL_22:
        v10 = *(_DWORD *)(a1 + 2328);
        v11 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2528) + (v7 & 0x7FFFFFFF));
        if ( v11 >= v10 )
          goto LABEL_33;
        v12 = *(_QWORD *)(a1 + 2320);
        while ( 1 )
        {
          v13 = v11;
          v14 = (_DWORD *)(v12 + 24LL * v11);
          if ( (v7 & 0x7FFFFFFF) == (*v14 & 0x7FFFFFFF) )
          {
            v15 = (unsigned int)v14[4];
            if ( (_DWORD)v15 != -1 && *(_DWORD *)(v12 + 24 * v15 + 20) == -1 )
              break;
          }
          v11 += 256;
          if ( v10 <= v11 )
            goto LABEL_33;
        }
        if ( v11 == -1 )
        {
LABEL_33:
          v17[0] = v7;
          v17[1] = 0;
          v18 = a2;
          result = sub_1E74F70(a1 + 2320, (__int64)v17);
        }
        else
        {
          while ( 1 )
          {
            result = v12 + 24 * v13;
            if ( *(_QWORD *)(result + 8) == a2 )
              break;
            v16 = *(_DWORD *)(result + 20);
            if ( v16 == -1 )
              goto LABEL_33;
            v13 = v16;
          }
        }
      }
LABEL_8:
      v3 += 40;
      if ( v5 == v3 )
        return result;
    }
    if ( (*(_DWORD *)v3 & 0xFFF00) == 0 || *(_BYTE *)(a1 + 914) )
      goto LABEL_8;
    v7 = *(_DWORD *)(v3 + 8);
LABEL_21:
    if ( v7 < 0 )
      goto LABEL_22;
    goto LABEL_8;
  }
  return result;
}

// Function: sub_2F413E0
// Address: 0x2f413e0
//
_QWORD *__fastcall sub_2F413E0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned int v7; // edx
  _QWORD *v8; // r9
  __int64 v9; // rax
  int v10; // ecx
  int v11; // ecx
  int v12; // ecx
  int v13; // ecx
  _QWORD *result; // rax
  __int64 v15; // rax
  unsigned int v16; // ecx
  int v17; // r9d
  __int64 v18; // rdx
  int v19; // edi

  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v4 <= 0 )
  {
LABEL_11:
    switch ( v5 )
    {
      case 2LL:
        result = a1;
        v16 = a3 & 0x1F;
        v18 = 4LL * (a3 >> 5);
        break;
      case 3LL:
        v15 = a3 >> 5;
        v16 = a3 & 0x1F;
        v17 = *(_DWORD *)(*a1 + 4 * v15);
        v18 = 4 * v15;
        result = a1;
        if ( !_bittest(&v17, a3) )
          return result;
        result = a1 + 1;
        break;
      case 1LL:
        LOBYTE(v16) = a3 & 0x1F;
        v18 = 4LL * (a3 >> 5);
LABEL_18:
        result = a1;
        if ( ((*(_DWORD *)(*a1 + v18) >> v16) & 1) != 0 )
          return (_QWORD *)a2;
        return result;
      default:
        return (_QWORD *)a2;
    }
    v19 = *(_DWORD *)(*result + v18);
    if ( !_bittest(&v19, v16) )
      return result;
    a1 = result + 1;
    goto LABEL_18;
  }
  v7 = a3 & 0x1F;
  v8 = &a1[4 * v4];
  v9 = 4LL * (a3 >> 5);
  while ( 1 )
  {
    v13 = *(_DWORD *)(*a1 + v9);
    if ( !_bittest(&v13, v7) )
      return a1;
    v10 = *(_DWORD *)(a1[1] + v9);
    if ( !_bittest(&v10, v7) )
      return a1 + 1;
    v11 = *(_DWORD *)(a1[2] + v9);
    if ( !_bittest(&v11, v7) )
      return a1 + 2;
    v12 = *(_DWORD *)(a1[3] + v9);
    if ( !_bittest(&v12, v7) )
      return a1 + 3;
    a1 += 4;
    if ( v8 == a1 )
    {
      v5 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}

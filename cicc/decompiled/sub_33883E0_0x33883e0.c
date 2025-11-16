// Function: sub_33883E0
// Address: 0x33883e0
//
__int64 __fastcall sub_33883E0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  __int64 v9; // rdi
  int v10; // edi
  unsigned int v11; // edx
  __int64 *v12; // rax
  char v13; // dl
  __int64 v14; // rax
  unsigned int v15; // r15d
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rax
  int v20; // eax
  int v21; // r10d
  char v22; // [rsp-41h] [rbp-41h] BYREF
  _BYTE *v23; // [rsp-40h] [rbp-40h] BYREF

  if ( *a2 <= 0x1Cu )
    return 1;
  v6 = a4;
  v23 = a2;
  if ( a3 )
  {
    if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
    {
      a5 = a3 + 16;
      a4 = a3 + 144;
      v10 = 7;
    }
    else
    {
      a5 = *(_QWORD *)(a3 + 16);
      v9 = *(unsigned int *)(a3 + 24);
      a4 = a5 + 16 * v9;
      if ( !(_DWORD)v9 )
        goto LABEL_8;
      v10 = v9 - 1;
    }
    v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(a5 + 16LL * v11);
    a6 = *v12;
    if ( a2 == (_BYTE *)*v12 )
    {
LABEL_7:
      if ( (__int64 *)a4 != v12 )
        return 1;
    }
    else
    {
      v20 = 1;
      while ( a6 != -4096 )
      {
        v21 = v20 + 1;
        v11 = v10 & (v20 + v11);
        v12 = (__int64 *)(a5 + 16LL * v11);
        a6 = *v12;
        if ( a2 == (_BYTE *)*v12 )
          goto LABEL_7;
        v20 = v21;
      }
    }
  }
LABEL_8:
  v22 = 0;
  sub_3388010(a1, (__int64 *)&v23, &v22, a4, a5, a6);
  if ( !v13 )
    return 1;
  v14 = (__int64)v23;
  if ( (*((_DWORD *)v23 + 1) & 0x7FFFFFF) == 0 )
    return 1;
  v15 = v6 + 1;
  v16 = 0;
  v17 = 32LL * (*((_DWORD *)v23 + 1) & 0x7FFFFFF);
  if ( (v23[7] & 0x40) == 0 )
    goto LABEL_15;
LABEL_11:
  v18 = *(_QWORD *)(v14 - 8);
  if ( v15 != 6 )
  {
    while ( (unsigned __int8)sub_33883E0(a1, *(_QWORD *)(v18 + v16), a3, v15) )
    {
      v16 += 32;
      if ( v17 == v16 )
        return 1;
      v14 = (__int64)v23;
      if ( (v23[7] & 0x40) != 0 )
        goto LABEL_11;
LABEL_15:
      v18 = v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
      if ( v15 == 6 )
        return 0;
    }
  }
  return 0;
}

// Function: sub_D19970
// Address: 0xd19970
//
__int64 __fastcall sub_D19970(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rcx
  int v3; // edx
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r13d
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // edx
  __int64 *v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // ebx
  int v21; // ecx
  int v22; // r9d

  v2 = *(_QWORD *)(*a2 + 8LL);
  v3 = *(unsigned __int8 *)(v2 + 8);
  if ( (unsigned int)(v3 - 17) <= 1 )
    LOBYTE(v3) = *(_BYTE *)(**(_QWORD **)(v2 + 16) + 8LL);
  if ( (_BYTE)v3 != 12 )
    return 0;
  v4 = a2[3];
  v9 = sub_D14860((unsigned __int8 *)v4);
  if ( (_BYTE)v9 )
    return 0;
  sub_D19710(a1, (__int64)a2, v5, v6, v7, v8);
  if ( !*(_BYTE *)(a1 + 380) )
  {
    if ( !sub_C8CA60(a1 + 352, (__int64)a2) )
      goto LABEL_14;
    return 1;
  }
  v11 = *(_QWORD **)(a1 + 360);
  v12 = &v11[*(unsigned int *)(a1 + 372)];
  if ( v11 != v12 )
  {
    while ( a2 != (_QWORD *)*v11 )
    {
      if ( v12 == ++v11 )
        goto LABEL_14;
    }
    return 1;
  }
LABEL_14:
  v13 = *(_QWORD *)(v4 + 8);
  v14 = *(unsigned __int8 *)(v13 + 8);
  if ( (unsigned int)(v14 - 17) <= 1 )
    LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
  if ( (_BYTE)v14 != 12 )
    return 0;
  v15 = *(unsigned int *)(a1 + 344);
  v16 = *(_QWORD *)(a1 + 328);
  if ( !(_DWORD)v15 )
    return v9;
  v17 = (v15 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v18 = (__int64 *)(v16 + 24LL * v17);
  v19 = *v18;
  if ( v4 == *v18 )
  {
LABEL_19:
    if ( v18 == (__int64 *)(v16 + 24 * v15) )
      return v9;
    v20 = *((_DWORD *)v18 + 4);
    if ( v20 <= 0x40 )
    {
      if ( v18[1] )
        return v9;
    }
    else if ( v20 != (unsigned int)sub_C444A0((__int64)(v18 + 1)) )
    {
      return v9;
    }
    return 1;
  }
  v21 = 1;
  while ( v19 != -4096 )
  {
    v22 = v21 + 1;
    v17 = (v15 - 1) & (v21 + v17);
    v18 = (__int64 *)(v16 + 24LL * v17);
    v19 = *v18;
    if ( v4 == *v18 )
      goto LABEL_19;
    v21 = v22;
  }
  return v9;
}

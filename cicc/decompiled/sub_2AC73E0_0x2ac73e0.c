// Function: sub_2AC73E0
// Address: 0x2ac73e0
//
__int64 __fastcall sub_2AC73E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 result; // rax
  int v6; // ebx
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 **v9; // rax
  bool v10; // zf
  __int64 *v11; // rax
  unsigned int v12; // esi
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // eax
  int v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v21; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 136);
  if ( *(_BYTE *)(a2 + 24) )
  {
    v8 = a2 + 16;
    result = sub_2AC70E0(*(_QWORD *)(a2 + 912), v4, a1, (unsigned int *)(a2 + 16), a2);
    if ( *(_BYTE *)(a2 + 12) )
    {
      if ( !*(_DWORD *)(a2 + 8) )
        return result;
    }
    else if ( *(_DWORD *)(a2 + 8) <= 1u )
    {
      return result;
    }
    result = sub_2C1B380(a1);
    if ( !(_BYTE)result )
      return result;
    if ( *(_DWORD *)(a2 + 16) || *(_BYTE *)(a2 + 20) )
      return sub_2BFBAC0(a2, a1 + 96, v8);
    v9 = (__int64 **)sub_BCE1B0(*(__int64 **)(v4 + 8), *(_QWORD *)(a2 + 8));
    v19 = sub_ACADE0(v9);
    v20 = a1 + 96;
    v10 = (unsigned __int8)sub_2ABE290(a2 + 32, &v20, &v21) == 0;
    v11 = v21;
    if ( !v10 )
    {
LABEL_19:
      v11[1] = v19;
      return sub_2BFBAC0(a2, a1 + 96, v8);
    }
    v12 = *(_DWORD *)(a2 + 56);
    v13 = *(_DWORD *)(a2 + 48);
    v22[0] = v21;
    ++*(_QWORD *)(a2 + 32);
    v14 = v13 + 1;
    if ( 4 * v14 >= 3 * v12 )
    {
      v12 *= 2;
    }
    else if ( v12 - *(_DWORD *)(a2 + 52) - v14 > v12 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a2 + 48) = v14;
      if ( *v11 != -4096 )
        --*(_DWORD *)(a2 + 52);
      v15 = v20;
      v11[1] = 0;
      *v11 = v15;
      goto LABEL_19;
    }
    sub_2AC6AB0(a2 + 32, v12);
    sub_2ABE290(a2 + 32, &v20, v22);
    v14 = *(_DWORD *)(a2 + 48) + 1;
    v11 = (__int64 *)v22[0];
    goto LABEL_16;
  }
  if ( *(_BYTE *)(a1 + 160) )
  {
    LODWORD(v22[0]) = 0;
    v16 = *(_QWORD *)(a2 + 912);
    BYTE4(v22[0]) = 0;
    return sub_2AC70E0(v16, v4, a1, (unsigned int *)v22, a2);
  }
  if ( *(_BYTE *)v4 == 62 && (unsigned __int8)sub_2AAA120(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL)) )
  {
    v16 = *(_QWORD *)(a2 + 912);
    v17 = *(_DWORD *)(a2 + 8) - 1;
    BYTE4(v22[0]) = *(_BYTE *)(a2 + 12);
    LODWORD(v22[0]) = v17;
    return sub_2AC70E0(v16, v4, a1, (unsigned int *)v22, a2);
  }
  result = *(unsigned int *)(a2 + 8);
  v18 = result;
  if ( (_DWORD)result )
  {
    v6 = 0;
    do
    {
      v7 = *(_QWORD *)(a2 + 912);
      LODWORD(v22[0]) = v6++;
      BYTE4(v22[0]) = 0;
      result = sub_2AC70E0(v7, v4, a1, (unsigned int *)v22, a2);
    }
    while ( v6 != v18 );
  }
  return result;
}

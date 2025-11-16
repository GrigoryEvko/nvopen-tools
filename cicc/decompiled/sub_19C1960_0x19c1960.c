// Function: sub_19C1960
// Address: 0x19c1960
//
__int64 __fastcall sub_19C1960(__int64 a1, unsigned __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  __int64 result; // rax
  __int64 v9; // r14
  _BOOL4 v10; // r8d
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r12
  _BOOL4 v19; // [rsp+4h] [rbp-3Ch]
  int v20; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v21[0] = a2;
  v7 = sub_19C18C0(a4, v21);
  result = 0;
  if ( !v6 )
    return result;
  v9 = v6;
  v10 = 1;
  if ( !v7 && v6 != a4 + 8 )
    v10 = v21[0] < *(_QWORD *)(v6 + 32);
  v19 = v10;
  v11 = sub_22077B0(40);
  *(_QWORD *)(v11 + 32) = v21[0];
  sub_220F040(v19, v11, v9, a4 + 8);
  ++*(_QWORD *)(a4 + 40);
  if ( !sub_1377F70(a1 + 56, v21[0]) )
  {
    if ( *a3 )
      return 0;
    *a3 = v21[0];
    return 1;
  }
  v12 = v21[0];
  v13 = sub_157EBA0(v21[0]);
  v14 = v13;
  if ( !v13 )
    goto LABEL_12;
  v15 = 0;
  v20 = sub_15F4D60(v13);
  if ( !v20 )
  {
LABEL_11:
    v12 = v21[0];
LABEL_12:
    v17 = *(_QWORD *)(v12 + 48);
    if ( v12 + 40 == v17 )
      return 1;
    while ( 1 )
    {
      v18 = v17 - 24;
      if ( !v17 )
        v18 = 0;
      if ( (unsigned __int8)sub_15F3040(v18) )
        break;
      if ( sub_15F3330(v18) )
        return 0;
      v17 = *(_QWORD *)(v17 + 8);
      if ( v12 + 40 == v17 )
        return 1;
    }
    return 0;
  }
  while ( 1 )
  {
    v16 = sub_15F4DF0(v14, v15);
    result = sub_19C1960(a1, v16, a3, a4);
    if ( !(_BYTE)result )
      return result;
    if ( v20 == ++v15 )
      goto LABEL_11;
  }
}

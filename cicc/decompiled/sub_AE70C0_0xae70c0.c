// Function: sub_AE70C0
// Address: 0xae70c0
//
bool __fastcall sub_AE70C0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  int v5; // eax
  unsigned __int8 **v7; // rax
  unsigned __int8 **v8; // rdx
  unsigned __int8 **v9; // rax
  __int64 v10; // rcx
  unsigned __int8 **v11; // rdx
  char v12; // dl
  _BYTE *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  _QWORD *v16; // r15
  _QWORD *v17; // r14
  unsigned __int8 **v18; // rax
  unsigned __int8 **v19; // rdx

  if ( !a3 )
    return 0;
  v5 = *a3;
  if ( (unsigned __int8)(v5 - 5) > 0x1Fu )
    return 0;
  if ( (_BYTE)v5 == 6 )
    return 1;
  if ( !*(_BYTE *)(a2 + 28) )
  {
    if ( !sub_C8CA60(a2, a3, (unsigned int)(v5 - 5), a4) )
      goto LABEL_12;
    return 1;
  }
  v7 = *(unsigned __int8 ***)(a2 + 8);
  v8 = &v7[*(unsigned int *)(a2 + 20)];
  if ( v7 != v8 )
  {
    while ( a3 != *v7 )
    {
      if ( v8 == ++v7 )
        goto LABEL_12;
    }
    return 1;
  }
LABEL_12:
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_19;
  v9 = *(unsigned __int8 ***)(a1 + 8);
  v10 = *(unsigned int *)(a1 + 20);
  v11 = &v9[v10];
  if ( v9 != v11 )
  {
    while ( a3 != *v9 )
    {
      if ( v11 == ++v9 )
        goto LABEL_30;
    }
    return 0;
  }
LABEL_30:
  if ( (unsigned int)v10 < *(_DWORD *)(a1 + 16) )
  {
    *(_DWORD *)(a1 + 20) = v10 + 1;
    *v11 = a3;
    ++*(_QWORD *)a1;
  }
  else
  {
LABEL_19:
    sub_C8CC70(a1, a3);
    if ( !v12 )
      return 0;
  }
  v13 = sub_A17150(a3 - 16);
  v16 = &v13[8 * v14];
  v17 = v13;
  if ( v13 != (_BYTE *)v16 )
  {
    do
    {
      if ( (unsigned __int8)sub_AE70C0(a1, a2, *v17) )
        sub_AE6EC0(a2, (__int64)a3);
      ++v17;
    }
    while ( v16 != v17 );
  }
  if ( *(_BYTE *)(a2 + 28) )
  {
    v18 = *(unsigned __int8 ***)(a2 + 8);
    v19 = &v18[*(unsigned int *)(a2 + 20)];
    if ( v18 == v19 )
      return 0;
    while ( a3 != *v18 )
    {
      if ( v19 == ++v18 )
        return 0;
    }
    return 1;
  }
  return sub_C8CA60(a2, a3, v14, v15) != 0;
}

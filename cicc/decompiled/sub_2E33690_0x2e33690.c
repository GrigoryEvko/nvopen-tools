// Function: sub_2E33690
// Address: 0x2e33690
//
void __fastcall sub_2E33690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // rcx
  __int64 *v9; // r12
  __int64 v10; // rsi
  _DWORD *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _DWORD *v16; // rax
  unsigned __int64 v17; // rdx
  int v18; // eax

  if ( a2 == a3 )
    return;
  v6 = *(__int64 **)(a1 + 112);
  v7 = *(unsigned int *)(a1 + 120);
  v8 = &v6[v7];
  v9 = v8;
  if ( v8 == v6 )
  {
LABEL_14:
    sub_2E32230(a2, a1);
    sub_2E32160(a3, a1, v12, v13, v14, v15);
    *v9 = a3;
    return;
  }
  v10 = (__int64)&v6[v7];
  do
  {
    while ( 1 )
    {
      if ( *v6 == a2 )
      {
        v9 = v6;
        if ( v8 != (__int64 *)v10 )
          goto LABEL_8;
        goto LABEL_5;
      }
      if ( *v6 == a3 )
        break;
LABEL_5:
      if ( v8 == ++v6 )
        goto LABEL_13;
    }
    v10 = (__int64)v6;
    if ( v9 != v8 )
      goto LABEL_8;
    ++v6;
  }
  while ( v8 != v6 );
LABEL_13:
  if ( (__int64 *)v10 == v8 )
    goto LABEL_14;
LABEL_8:
  if ( *(_QWORD *)(a1 + 152) != *(_QWORD *)(a1 + 144) )
  {
    v11 = (_DWORD *)sub_2E32F70(a1, v10);
    if ( *v11 != -1 )
    {
      v16 = (_DWORD *)sub_2E32F70(a1, (__int64)v9);
      v17 = (unsigned int)*v16 + (unsigned __int64)(unsigned int)*v11;
      v18 = *v11 + *v16;
      if ( v17 > 0x80000000 )
        v18 = 0x80000000;
      *v11 = v18;
    }
  }
  sub_2E33590(a1, v9, 0);
}

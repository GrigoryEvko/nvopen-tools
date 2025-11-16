// Function: sub_D88EE0
// Address: 0xd88ee0
//
__int64 __fastcall sub_D88EE0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4, __int64 a5)
{
  _QWORD *v7; // rax
  _QWORD *v8; // r9
  _QWORD *v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 v18; // rax
  _QWORD *v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // eax
  unsigned int v24; // eax

  v7 = *(_QWORD **)(a2 + 16);
  if ( !v7 )
    goto LABEL_8;
  v8 = (_QWORD *)(a2 + 8);
  v12 = (_QWORD *)(a2 + 8);
  do
  {
    while ( 1 )
    {
      v13 = v7[2];
      v14 = v7[3];
      if ( v7[4] >= a3 )
        break;
      v7 = (_QWORD *)v7[3];
      if ( !v14 )
        goto LABEL_6;
    }
    v12 = v7;
    v7 = (_QWORD *)v7[2];
  }
  while ( v13 );
LABEL_6:
  if ( v8 == v12 )
    goto LABEL_8;
  if ( v12[4] > a3 )
    goto LABEL_8;
  v18 = v12[13];
  if ( !v18 )
    goto LABEL_8;
  v19 = v12 + 12;
  do
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v18 + 16);
      v21 = *(_QWORD *)(v18 + 24);
      if ( *(_DWORD *)(v18 + 32) >= a4 )
        break;
      v18 = *(_QWORD *)(v18 + 24);
      if ( !v21 )
        goto LABEL_17;
    }
    v19 = (_QWORD *)v18;
    v18 = *(_QWORD *)(v18 + 16);
  }
  while ( v20 );
LABEL_17:
  if ( v12 + 12 == v19 || *((_DWORD *)v19 + 8) > a4 )
  {
LABEL_8:
    v15 = *(_DWORD *)(a2 + 56);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 > 0x40 )
    {
      sub_C43780(a1, (const void **)(a2 + 48));
      v24 = *(_DWORD *)(a2 + 72);
      *(_DWORD *)(a1 + 24) = v24;
      if ( v24 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)(a2 + 48);
      v16 = *(_DWORD *)(a2 + 72);
      *(_DWORD *)(a1 + 24) = v16;
      if ( v16 <= 0x40 )
      {
LABEL_10:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 64);
        return a1;
      }
    }
    sub_C43780(a1 + 16, (const void **)(a2 + 64));
    return a1;
  }
  if ( sub_AAF7D0((__int64)(v19 + 5)) )
  {
    v22 = *((_DWORD *)v19 + 12);
    *(_DWORD *)(a1 + 8) = v22;
    if ( v22 > 0x40 )
      sub_C43780(a1, (const void **)v19 + 5);
    else
      *(_QWORD *)a1 = v19[5];
    v23 = *((_DWORD *)v19 + 16);
    *(_DWORD *)(a1 + 24) = v23;
    if ( v23 > 0x40 )
      sub_C43780(a1 + 16, (const void **)v19 + 7);
    else
      *(_QWORD *)(a1 + 16) = v19[7];
  }
  else if ( sub_AAF760((__int64)(v19 + 5)) )
  {
    sub_AAF450(a1, a2 + 48);
  }
  else
  {
    sub_D87200(a1, (__int64)(v19 + 5), a5);
  }
  return a1;
}

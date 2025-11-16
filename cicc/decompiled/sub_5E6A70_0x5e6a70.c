// Function: sub_5E6A70
// Address: 0x5e6a70
//
void __fastcall sub_5E6A70(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned int v10; // esi
  unsigned int v11; // ecx
  _QWORD *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v8 = *a2;
  if ( !*a2 || (v9 = *(_QWORD *)(v8 + 8), v10 = *(_DWORD *)(a3 + 44), v11 = *(_DWORD *)(v9 + 44), v11 > v10) )
  {
    v8 = *a1;
    if ( !*a1 )
    {
      v12 = 0;
      goto LABEL_18;
    }
    v9 = *(_QWORD *)(v8 + 8);
    v10 = *(_DWORD *)(a3 + 44);
    v11 = *(_DWORD *)(v9 + 44);
  }
  v12 = 0;
  while ( v10 >= v11 )
  {
    if ( a3 == v9 && *(_QWORD *)(v8 + 16) == a5 )
      goto LABEL_9;
    if ( !*(_QWORD *)v8 )
    {
      v12 = (_QWORD *)v8;
      break;
    }
    v9 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
    v12 = (_QWORD *)v8;
    v8 = *(_QWORD *)v8;
    v11 = *(_DWORD *)(v9 + 44);
  }
LABEL_18:
  v8 = qword_4CF7FD8;
  if ( qword_4CF7FD8 )
  {
    qword_4CF7FD8 = *(_QWORD *)qword_4CF7FD8;
  }
  else
  {
    v18 = a5;
    v20 = a4;
    v17 = sub_823970(48);
    a5 = v18;
    a4 = v20;
    v8 = v17;
  }
  *(_QWORD *)v8 = 0;
  *(_QWORD *)(v8 + 24) = 0;
  *(_QWORD *)(v8 + 32) = 0;
  *(_QWORD *)(v8 + 40) = 0;
  *(_QWORD *)(v8 + 8) = a3;
  *(_QWORD *)(v8 + 16) = a5;
  if ( *(_BYTE *)(a3 + 80) == 17 )
  {
    v14 = *(_QWORD *)(a3 + 88);
    v15 = 0;
    while ( v14 )
    {
      v16 = v14;
      v14 = *(_QWORD *)(v14 + 8);
      if ( *(_BYTE *)(v16 + 80) == 10 )
        v15 -= ((*(_BYTE *)(*(_QWORD *)(v16 + 88) + 192LL) & 2) == 0) - 1;
    }
    *(_DWORD *)(v8 + 40) = v15;
  }
  else
  {
    *(_DWORD *)(v8 + 40) = 1;
  }
  if ( *a1 )
  {
    if ( v12 )
    {
      *(_QWORD *)v8 = *v12;
      *v12 = v8;
      goto LABEL_9;
    }
    *(_QWORD *)v8 = *a1;
  }
  *a1 = v8;
LABEL_9:
  v19 = a4;
  if ( a4 )
  {
    v13 = sub_878440();
    *(_QWORD *)(v13 + 8) = v19;
    if ( *(_QWORD *)(v8 + 24) )
      **(_QWORD **)(v8 + 32) = v13;
    else
      *(_QWORD *)(v8 + 24) = v13;
    *(_QWORD *)(v8 + 32) = v13;
  }
  else
  {
    ++*(_DWORD *)(v8 + 44);
  }
  *a2 = v8;
}

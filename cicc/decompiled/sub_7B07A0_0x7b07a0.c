// Function: sub_7B07A0
// Address: 0x7b07a0
//
__int64 __fastcall sub_7B07A0(__int64 a1, unsigned __int8 *a2, int a3)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  int v5; // eax
  int v6; // r15d
  __int64 v7; // r11
  unsigned int i; // r12d
  char *v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 *v13; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  char *v16; // rax
  unsigned __int8 *v17; // r15
  _BOOL4 v18; // eax
  __int64 v19; // r13
  __int64 v20; // rax
  unsigned int v21; // eax
  int v22; // edi
  __int64 v23; // r8
  __int64 *v24; // rcx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v29; // [rsp+8h] [rbp-48h]

  v3 = (__int64 *)unk_4D048E0;
  v4 = sub_722DF0(a2);
  v5 = sub_887620(v4);
  v6 = *((_DWORD *)v3 + 2);
  v7 = *v3;
  for ( i = v6 & v5; ; i = v6 & (i + 1) )
  {
    v10 = v7 + 16LL * i;
    v9 = *(char **)v10;
    if ( !*(_QWORD *)v10 || !a2 )
      break;
    v29 = v7;
    if ( !sub_722E50(v9, (char *)a2, 0, 0, 1) )
      goto LABEL_9;
    v9 = *(char **)v10;
    v7 = v29;
LABEL_5:
    if ( !v9 )
      goto LABEL_11;
  }
  if ( a2 != (unsigned __int8 *)v9 )
    goto LABEL_5;
LABEL_9:
  v11 = *(_QWORD *)(v10 + 8);
  if ( v11 )
    return v11;
LABEL_11:
  if ( a3 )
    v13 = (__int64 *)qword_4D048D0;
  else
    v13 = (__int64 *)unk_4D048D8;
  v14 = *v13;
  v15 = *v13 + 16LL * (unsigned int)(*((_DWORD *)v13 + 2) + 1);
  if ( v15 == *v13 )
    return 0;
  while ( 1 )
  {
    if ( *(_QWORD *)v14 )
    {
      v16 = sub_7B06F0(*(char **)v14, a3, 0, 1);
      v17 = (unsigned __int8 *)v16;
      if ( v16 && a2 )
        v18 = !sub_722E50((char *)a2, v16, 0, 0, 1);
      else
        v18 = a2 == (unsigned __int8 *)v16;
      if ( v18 )
        break;
    }
    v14 += 16;
    if ( v15 == v14 )
      return 0;
  }
  v11 = *(_QWORD *)(v14 + 8);
  v19 = unk_4D048E0;
  v20 = sub_722DF0(v17);
  v21 = sub_887620(v20);
  v22 = *(_DWORD *)(v19 + 8);
  v23 = v22 & v21;
  v24 = (__int64 *)(*(_QWORD *)v19 + 16 * v23);
  if ( *v24 )
  {
    v25 = v22 & v21;
    do
    {
      v25 = v22 & (v25 + 1);
      v26 = (__int64 *)(*(_QWORD *)v19 + 16LL * v25);
    }
    while ( *v26 );
    v27 = *v24;
    *v26 = *v24;
    if ( v27 )
      v26[1] = v24[1];
    *v24 = 0;
  }
  sub_7ADAC0((__int64 *)v19, v23, (__int64)v17, v11);
  return v11;
}

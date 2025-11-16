// Function: sub_C80AD0
// Address: 0xc80ad0
//
__int64 __fastcall sub_C80AD0(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r14
  unsigned int v5; // esi
  char *v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  char *v11; // rdx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdi

  v2 = sub_C80650(*(char **)a1, *(_QWORD *)(a1 + 8), *(_DWORD *)(a1 + 40));
  v3 = *(_QWORD *)(a1 + 32);
  v4 = v2;
  while ( 1 )
  {
    v5 = *(_DWORD *)(a1 + 40);
    v6 = *(char **)a1;
    if ( !v3 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = 0;
      if ( *(_QWORD *)(a1 + 32) != v7 )
        goto LABEL_12;
      if ( !v7 )
        goto LABEL_8;
LABEL_24:
      if ( sub_C80220(*(_BYTE *)(*(_QWORD *)a1 + v7 - 1), v5) && (v4 == -1 || v3 - 1 > v4) )
      {
        --*(_QWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 16) = ".";
        *(_QWORD *)(a1 + 24) = 1;
        return a1;
      }
      v7 = *(_QWORD *)(a1 + 8);
      v5 = *(_DWORD *)(a1 + 40);
      goto LABEL_9;
    }
    if ( v3 - 1 == v4 )
      goto LABEL_18;
    if ( !sub_C80220(v6[v3 - 1], v5) )
      break;
    --v3;
  }
  v5 = *(_DWORD *)(a1 + 40);
LABEL_18:
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 32) == v7 )
  {
    if ( v7 )
      goto LABEL_24;
LABEL_8:
    v7 = *(_QWORD *)(a1 + 8);
  }
LABEL_9:
  v6 = *(char **)a1;
  if ( v7 > v3 )
    v7 = v3;
  v8 = v7;
LABEL_12:
  v9 = sub_C80770(v6, v8, v5);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(char **)a1;
  v12 = v10;
  if ( v9 <= v10 )
    v12 = v9;
  v13 = 0;
  if ( v12 <= v3 )
  {
    if ( v10 <= v3 )
      v3 = *(_QWORD *)(a1 + 8);
    v13 = v3 - v12;
  }
  *(_QWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 16) = &v11[v12];
  *(_QWORD *)(a1 + 32) = v9;
  return a1;
}

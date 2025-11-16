// Function: sub_30B1A90
// Address: 0x30b1a90
//
__int64 __fastcall sub_30B1A90(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *i; // r13
  unsigned __int64 v5; // r15
  __int64 v6; // rdi
  _BYTE *v7; // rax
  _BYTE *v8; // rax

  v3 = *(__int64 **)(a2 + 96);
  for ( i = &v3[*(unsigned int *)(a2 + 104)]; i != v3; ++*(_QWORD *)(v6 + 32) )
  {
    while ( 1 )
    {
      v5 = *v3;
      if ( !sub_30B1A10(a2, *v3) )
        break;
LABEL_3:
      if ( i == ++v3 )
        goto LABEL_7;
    }
    v6 = sub_30B0F90(a1, v5);
    v7 = *(_BYTE **)(v6 + 32);
    if ( *(_BYTE **)(v6 + 24) == v7 )
    {
      sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
      goto LABEL_3;
    }
    ++v3;
    *v7 = 10;
  }
LABEL_7:
  v8 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v8 )
  {
    sub_CB6200(a1, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v8 = 10;
    ++*(_QWORD *)(a1 + 32);
  }
  return a1;
}

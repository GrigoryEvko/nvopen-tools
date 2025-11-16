// Function: sub_C414D0
// Address: 0xc414d0
//
__int64 __fastcall sub_C414D0(__int64 a1)
{
  __int64 v1; // rbx
  void *v2; // r12
  __int64 v3; // rax
  char v5; // al
  void *v6; // rdx
  unsigned __int8 v7; // al
  char v8; // al
  void *v9; // rdx
  char v10; // al

  v1 = a1;
  v2 = sub_C33340();
  if ( *(void **)a1 == v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(v3 + 20) & 7) == 3 )
      return (*(_BYTE *)(v3 + 20) & 8) == 0 ? 64 : 32;
    v5 = sub_C40310(a1);
  }
  else
  {
    v3 = a1;
    if ( (*(_BYTE *)(a1 + 20) & 7) == 3 )
      return (*(_BYTE *)(v3 + 20) & 8) == 0 ? 64 : 32;
    v5 = sub_C33940(a1);
  }
  v6 = *(void **)a1;
  if ( v5 )
  {
    if ( v2 != v6 )
    {
LABEL_9:
      v8 = sub_C33940(a1);
      goto LABEL_10;
    }
  }
  else
  {
    if ( v2 != v6 )
    {
      v7 = *(_BYTE *)(a1 + 20) & 7;
      if ( v7 == 3 || v7 <= 1u )
        goto LABEL_9;
      return (*(_BYTE *)(v1 + 20) & 8) == 0 ? 256 : 8;
    }
    v10 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 20LL) & 7;
    if ( v10 != 1 && v10 != 3 && v10 )
    {
      v1 = *(_QWORD *)(a1 + 8);
      return (*(_BYTE *)(v1 + 20) & 8) == 0 ? 256 : 8;
    }
  }
  v8 = sub_C40310(a1);
LABEL_10:
  v9 = *(void **)a1;
  if ( !v8 )
  {
    if ( v2 == v9 )
    {
      a1 = *(_QWORD *)(a1 + 8);
      if ( (*(_BYTE *)(a1 + 20) & 7) == 0 )
        return (*(_BYTE *)(a1 + 20) & 8) == 0 ? 512 : 4;
    }
    else if ( (*(_BYTE *)(a1 + 20) & 7) == 0 )
    {
      return (*(_BYTE *)(a1 + 20) & 8) == 0 ? 512 : 4;
    }
    return (unsigned int)((unsigned __int8)sub_C35FD0((_BYTE *)a1) == 0) + 1;
  }
  if ( v2 == v9 )
    v1 = *(_QWORD *)(a1 + 8);
  return (*(_BYTE *)(v1 + 20) & 8) == 0 ? 128 : 16;
}

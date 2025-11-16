// Function: sub_D9ABE0
// Address: 0xd9abe0
//
__int64 __fastcall sub_D9ABE0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  void *v5; // rdx
  int v6; // edx
  __int64 result; // rax

  v3 = sub_CB69B0(a2, a3);
  v4 = sub_D9ABD0(a1);
  sub_D955C0(v4, v3);
  v5 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v5 <= 0xDu )
  {
    sub_CB6200(v3, " Added Flags: ", 0xEu);
    v6 = *(_DWORD *)(a1 + 48);
    result = *(_QWORD *)(a2 + 32);
    if ( (v6 & 1) == 0 )
    {
LABEL_3:
      if ( (v6 & 2) == 0 )
        goto LABEL_4;
LABEL_10:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 5 )
      {
        sub_CB6200(a2, "<nssw>", 6u);
        result = *(_QWORD *)(a2 + 32);
      }
      else
      {
        *(_DWORD *)result = 1936944700;
        *(_WORD *)(result + 4) = 15991;
        result = *(_QWORD *)(a2 + 32) + 6LL;
        *(_QWORD *)(a2 + 32) = result;
      }
      goto LABEL_4;
    }
  }
  else
  {
    qmemcpy(v5, " Added Flags: ", 14);
    *(_QWORD *)(v3 + 32) += 14LL;
    v6 = *(_DWORD *)(a1 + 48);
    result = *(_QWORD *)(a2 + 32);
    if ( (v6 & 1) == 0 )
      goto LABEL_3;
  }
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - result) <= 5 )
  {
    sub_CB6200(a2, "<nusw>", 6u);
    result = *(_QWORD *)(a2 + 32);
  }
  else
  {
    *(_DWORD *)result = 1937075772;
    *(_WORD *)(result + 4) = 15991;
    result = *(_QWORD *)(a2 + 32) + 6LL;
    *(_QWORD *)(a2 + 32) = result;
  }
  if ( (*(_DWORD *)(a1 + 48) & 2) != 0 )
    goto LABEL_10;
LABEL_4:
  if ( *(_QWORD *)(a2 + 24) == result )
    return sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *(_BYTE *)result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}

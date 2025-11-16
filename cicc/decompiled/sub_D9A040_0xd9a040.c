// Function: sub_D9A040
// Address: 0xd9a040
//
__int64 __fastcall sub_D9A040(__int64 a1, int a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdx
  void *v5; // rdx

  switch ( a2 )
  {
    case 1:
      v3 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v3) <= 8 )
      {
        sub_CB6200(a1, "Invariant", 9u);
      }
      else
      {
        *(_BYTE *)(v3 + 8) = 116;
        *(_QWORD *)v3 = 0x6E61697261766E49LL;
        *(_QWORD *)(a1 + 32) += 9LL;
      }
      break;
    case 2:
      v5 = *(void **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v5 <= 9u )
      {
        sub_CB6200(a1, "Computable", 0xAu);
      }
      else
      {
        qmemcpy(v5, "Computable", 10);
        *(_QWORD *)(a1 + 32) += 10LL;
      }
      break;
    case 0:
      v4 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v4) <= 6 )
      {
        sub_CB6200(a1, (unsigned __int8 *)"Variant", 7u);
      }
      else
      {
        *(_DWORD *)v4 = 1769103702;
        *(_WORD *)(v4 + 4) = 28257;
        *(_BYTE *)(v4 + 6) = 116;
        *(_QWORD *)(a1 + 32) += 7LL;
      }
      break;
  }
  return a1;
}

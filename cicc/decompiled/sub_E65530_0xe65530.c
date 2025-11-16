// Function: sub_E65530
// Address: 0xe65530
//
void __fastcall sub_E65530(__int64 a1, char **a2)
{
  __int64 v2; // r13
  __int64 i; // rbx

  v2 = *(_QWORD *)(a1 + 1680);
  for ( i = v2 + ((unsigned __int64)*(unsigned int *)(a1 + 1688) << 6); v2 != i; i -= 64 )
  {
    if ( (unsigned __int8)sub_C84AB0(
                            a2,
                            *(char **)(i - 64),
                            *(_QWORD *)(i - 56),
                            *(const void **)(i - 32),
                            *(_QWORD *)(i - 24),
                            0) )
      break;
  }
}

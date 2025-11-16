// Function: sub_2D9F520
// Address: 0x2d9f520
//
__int64 __fastcall sub_2D9F520(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  unsigned __int64 v8; // rdx

  v6 = *a2;
  if ( !*a2 )
  {
    *(_BYTE *)a1 = 0;
    result = *(unsigned int *)(a1 + 48);
    v6 = *a2;
    v8 = result + 1;
    if ( result + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 52) )
      goto LABEL_3;
LABEL_5:
    sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v8, 8u, a5, a6);
    result = *(unsigned int *)(a1 + 48);
    goto LABEL_3;
  }
  result = *(unsigned int *)(a1 + 48);
  v8 = result + 1;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
    goto LABEL_5;
LABEL_3:
  *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * result) = v6;
  ++*(_DWORD *)(a1 + 48);
  return result;
}

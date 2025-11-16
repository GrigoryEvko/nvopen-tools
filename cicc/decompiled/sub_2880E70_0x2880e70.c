// Function: sub_2880E70
// Address: 0x2880e70
//
__int64 __fastcall sub_2880E70(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax

  if ( !*((_DWORD *)a1 + 2) )
    a5 = *a1;
  v6 = *(unsigned int *)(a2 + 40);
  v7 = (unsigned int)(a5 - v6);
  if ( a3 )
    v8 = v7 * a3;
  else
    v8 = v7 * *(unsigned int *)(a2 + 20);
  return v6 + v8;
}

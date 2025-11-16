// Function: sub_3373A60
// Address: 0x3373a60
//
__int64 __fastcall sub_3373A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // r14
  __int64 v8; // rbx
  const void *v9; // r15
  __int64 v10; // rdi

  v6 = (unsigned int *)(a1 + 416);
  v8 = *(unsigned int *)(a1 + 712);
  v9 = *(const void **)(a1 + 704);
  v10 = *(unsigned int *)(a1 + 424);
  if ( v8 + v10 > (unsigned __int64)*(unsigned int *)(a1 + 428) )
  {
    sub_C8D5F0((__int64)v6, (const void *)(a1 + 432), v8 + v10, 0x10u, a5, a6);
    v10 = *(unsigned int *)(a1 + 424);
  }
  if ( 16 * v8 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 416) + 16 * v10), v9, 16 * v8);
    LODWORD(v10) = *(_DWORD *)(a1 + 424);
  }
  *(_DWORD *)(a1 + 712) = 0;
  *(_DWORD *)(a1 + 424) = v8 + v10;
  return sub_33736A0((__int64 *)a1, v6);
}

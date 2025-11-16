// Function: sub_16E7790
// Address: 0x16e7790
//
void __fastcall sub_16E7790(__int64 a1, const void *a2, size_t a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  __int64 v8; // rdi

  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(unsigned int *)(v7 + 8);
  if ( a3 > (unsigned __int64)*(unsigned int *)(v7 + 12) - v8 )
  {
    sub_16CD150(v7, (const void *)(v7 + 16), v8 + a3, 1, a5, a6);
    v8 = *(unsigned int *)(v7 + 8);
  }
  if ( a3 )
  {
    memcpy((void *)(*(_QWORD *)v7 + v8), a2, a3);
    LODWORD(v8) = *(_DWORD *)(v7 + 8);
  }
  *(_DWORD *)(v7 + 8) = v8 + a3;
}

// Function: sub_16CD150
// Address: 0x16cd150
//
void __fastcall sub_16CD150(__int64 a1, const void *a2, unsigned __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // rbx
  const void *v8; // r15
  unsigned __int64 v9; // rdx
  void *v10; // r13
  __int64 v11; // rax

  v7 = a3;
  if ( a3 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v8 = *(const void **)a1;
  if ( 2 * (unsigned __int64)*(unsigned int *)(a1 + 12) + 1 >= v7 )
    v7 = 2LL * *(unsigned int *)(a1 + 12) + 1;
  if ( v7 > 0xFFFFFFFF )
    v7 = 0xFFFFFFFFLL;
  v9 = v7 * a4;
  if ( v8 == a2 )
  {
    v10 = (void *)malloc(v9);
    if ( !v10 )
    {
      if ( v7 * a4 || (v11 = malloc(1u)) == 0 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v8 = *(const void **)a1;
      }
      else
      {
        v10 = (void *)v11;
      }
    }
    memcpy(v10, v8, a4 * *(unsigned int *)(a1 + 8));
  }
  else
  {
    v10 = realloc(*(_QWORD *)a1, v9, v9, a4, a5, a6);
    if ( !v10 && (v7 * a4 || (v10 = (void *)malloc(1u)) == 0) )
      sub_16BD1C0("Allocation failed", 1u);
  }
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v7;
}

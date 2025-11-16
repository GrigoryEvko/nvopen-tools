// Function: sub_1BFC1A0
// Address: 0x1bfc1a0
//
void __fastcall sub_1BFC1A0(__int64 a1, int a2, unsigned __int8 a3)
{
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  size_t v6; // rdx
  char *v7; // r14
  int v8; // ecx
  unsigned int v9; // ebx
  int v10; // ecx
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  size_t n; // [rsp+8h] [rbp-38h]

  v4 = (unsigned int)(a2 + 63) >> 6;
  *(_DWORD *)(a1 + 16) = a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v5 = malloc(8 * v4);
  v6 = 8 * v4;
  v7 = (char *)v5;
  if ( !v5 )
  {
    if ( 8 * v4 || (v12 = malloc(1u), v6 = 0, !v12) )
    {
      n = v6;
      sub_16BD1C0("Allocation failed", 1u);
      v6 = n;
    }
    else
    {
      v7 = (char *)v12;
    }
  }
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = v4;
  if ( (unsigned int)(a2 + 63) >> 6 )
  {
    memset(v7, -a3, v6);
    if ( !a3 )
      return;
    v8 = *(_DWORD *)(a1 + 16);
    v9 = (unsigned int)(v8 + 63) >> 6;
    if ( v4 > v9 )
    {
      v11 = v4 - v9;
      if ( v11 )
      {
        memset(&v7[8 * v9], 0, 8 * v11);
        v8 = *(_DWORD *)(a1 + 16);
      }
    }
  }
  else
  {
    if ( !a3 )
      return;
    v8 = *(_DWORD *)(a1 + 16);
    v9 = (unsigned int)(v8 + 63) >> 6;
  }
  v10 = v8 & 0x3F;
  if ( v10 )
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v9 - 1)) &= ~(-1LL << v10);
}

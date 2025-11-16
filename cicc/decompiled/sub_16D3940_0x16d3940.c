// Function: sub_16D3940
// Address: 0x16d3940
//
void *__fastcall sub_16D3940(void ***a1, const void *a2, size_t a3)
{
  unsigned __int64 v3; // r14
  void **v5; // rbx
  char *v6; // r12
  _BYTE *v7; // rax
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  int v10; // r9d
  unsigned __int64 v11; // r8
  __int64 v12; // rdx
  void *result; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  _QWORD *v17; // rax
  unsigned int v18; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v19; // [rsp+8h] [rbp-38h]
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = a3 + 1;
  v5 = *a1;
  v6 = (char *)**a1;
  v7 = (*a1)[1];
  (*a1)[10] = (char *)(*a1)[10] + a3 + 1;
  if ( a3 + 1 <= v7 - v6 )
  {
LABEL_10:
    *v5 = &v6[v3];
    goto LABEL_11;
  }
  if ( v3 <= 0x1000 )
  {
    v8 = 0x40000000000LL;
    v18 = *((_DWORD *)v5 + 6);
    if ( v18 >> 7 < 0x1E )
      v8 = 4096LL << (v18 >> 7);
    v19 = v8;
    v9 = malloc(v8);
    v11 = v19;
    v12 = v18;
    v6 = (char *)v9;
    if ( !v9 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v12 = *((unsigned int *)v5 + 6);
      v11 = v19;
    }
    if ( *((_DWORD *)v5 + 7) <= (unsigned int)v12 )
    {
      v20 = v11;
      sub_16CD150((__int64)(v5 + 2), v5 + 4, 0, 8, v11, v10);
      v12 = *((unsigned int *)v5 + 6);
      v11 = v20;
    }
    *((_QWORD *)v5[2] + v12) = v6;
    ++*((_DWORD *)v5 + 6);
    v5[1] = &v6[v11];
    goto LABEL_10;
  }
  v6 = (char *)malloc(a3 + 1);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v16 = *((unsigned int *)v5 + 18);
  if ( (unsigned int)v16 >= *((_DWORD *)v5 + 19) )
  {
    sub_16CD150((__int64)(v5 + 8), v5 + 10, 0, 16, v14, v15);
    v16 = *((unsigned int *)v5 + 18);
  }
  v17 = (char *)v5[8] + 16 * v16;
  *v17 = v6;
  v17[1] = v3;
  ++*((_DWORD *)v5 + 18);
LABEL_11:
  result = memcpy(v6, a2, a3);
  v6[a3] = 0;
  return result;
}

// Function: sub_18DB490
// Address: 0x18db490
//
void *__fastcall sub_18DB490(__int64 a1)
{
  void *result; // rax
  void *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  void *v6; // rdi
  unsigned int v7; // eax
  __int64 v8; // rdx

  result = 0;
  *(_WORD *)a1 = 0;
  ++*(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0;
  v3 = *(void **)(a1 + 32);
  if ( v3 == *(void **)(a1 + 24) )
    goto LABEL_6;
  v4 = 4 * (*(_DWORD *)(a1 + 44) - *(_DWORD *)(a1 + 48));
  v5 = *(unsigned int *)(a1 + 40);
  if ( v4 < 0x20 )
    v4 = 32;
  if ( (unsigned int)v5 <= v4 )
  {
    result = memset(v3, -1, 8 * v5);
LABEL_6:
    *(_QWORD *)(a1 + 44) = 0;
    goto LABEL_7;
  }
  result = sub_16CC920(a1 + 16);
LABEL_7:
  ++*(_QWORD *)(a1 + 72);
  v6 = *(void **)(a1 + 88);
  if ( v6 != *(void **)(a1 + 80) )
  {
    v7 = 4 * (*(_DWORD *)(a1 + 100) - *(_DWORD *)(a1 + 104));
    v8 = *(unsigned int *)(a1 + 96);
    if ( v7 < 0x20 )
      v7 = 32;
    if ( (unsigned int)v8 > v7 )
    {
      result = sub_16CC920(a1 + 72);
      goto LABEL_13;
    }
    result = memset(v6, -1, 8 * v8);
  }
  *(_QWORD *)(a1 + 100) = 0;
LABEL_13:
  *(_BYTE *)(a1 + 128) = 0;
  return result;
}

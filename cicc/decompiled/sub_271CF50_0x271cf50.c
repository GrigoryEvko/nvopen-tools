// Function: sub_271CF50
// Address: 0x271cf50
//
void *__fastcall sub_271CF50(__int64 a1, __int64 a2)
{
  void *result; // rax
  bool v3; // zf
  unsigned int v4; // eax
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rdx

  result = 0;
  ++*(_QWORD *)(a1 + 16);
  v3 = *(_BYTE *)(a1 + 44) == 0;
  *(_WORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  if ( !v3 )
    goto LABEL_6;
  v4 = 4 * (*(_DWORD *)(a1 + 36) - *(_DWORD *)(a1 + 40));
  v5 = *(unsigned int *)(a1 + 32);
  if ( v4 < 0x20 )
    v4 = 32;
  if ( (unsigned int)v5 <= v4 )
  {
    a2 = 0xFFFFFFFFLL;
    result = memset(*(void **)(a1 + 24), -1, 8 * v5);
LABEL_6:
    *(_QWORD *)(a1 + 36) = 0;
    goto LABEL_7;
  }
  result = sub_C8C990(a1 + 16, a2);
LABEL_7:
  ++*(_QWORD *)(a1 + 64);
  if ( !*(_BYTE *)(a1 + 92) )
  {
    v6 = 4 * (*(_DWORD *)(a1 + 84) - *(_DWORD *)(a1 + 88));
    v7 = *(unsigned int *)(a1 + 80);
    if ( v6 < 0x20 )
      v6 = 32;
    if ( (unsigned int)v7 > v6 )
    {
      result = sub_C8C990(a1 + 64, a2);
      goto LABEL_13;
    }
    result = memset(*(void **)(a1 + 72), -1, 8 * v7);
  }
  *(_QWORD *)(a1 + 84) = 0;
LABEL_13:
  *(_BYTE *)(a1 + 112) = 0;
  return result;
}

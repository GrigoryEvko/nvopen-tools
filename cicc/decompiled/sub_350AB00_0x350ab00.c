// Function: sub_350AB00
// Address: 0x350ab00
//
unsigned __int64 __fastcall sub_350AB00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // r14d
  unsigned __int64 v8; // r12
  unsigned __int64 result; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // [rsp+8h] [rbp-28h]

  v6 = *a3;
  v7 = *(_DWORD *)(a1 + 72);
  a3[10] += 16;
  v8 = (v6 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( a3[1] >= v8 + 16 && v6 )
  {
    *a3 = v8 + 16;
    result = (v6 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( !v8 )
      goto LABEL_5;
  }
  else
  {
    result = sub_9D1E70((__int64)a3, 16, 16, 4);
    v8 = result;
  }
  *(_DWORD *)result = v7;
  *(_QWORD *)(result + 8) = a2;
LABEL_5:
  v10 = *(unsigned int *)(a1 + 72);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
  {
    v11 = result;
    sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v10 + 1, 8u, v10 + 1, a6);
    v10 = *(unsigned int *)(a1 + 72);
    result = v11;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v10) = v8;
  ++*(_DWORD *)(a1 + 72);
  return result;
}

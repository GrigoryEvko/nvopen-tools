// Function: sub_BCBD60
// Address: 0xbcbd60
//
unsigned __int64 __fastcall sub_BCBD60(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        const void *a5,
        __int64 a6,
        void *src,
        __int64 a8)
{
  __int64 v11; // rax
  char *v12; // rdi
  size_t v13; // r12
  __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned __int64 result; // rax

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 20;
  *(_QWORD *)(a1 + 16) = 0;
  v11 = sub_C94910(*a2 + 2736LL, a3, a4);
  *(_DWORD *)(a1 + 12) = a6;
  v12 = (char *)(a1 + 48);
  v13 = 8 * a6;
  *(_QWORD *)(a1 + 24) = v11;
  *(_QWORD *)(a1 + 32) = v14;
  *(_QWORD *)(a1 + 16) = a1 + 48;
  if ( v13 )
    v12 = (char *)memcpy(v12, a5, v13) + v13;
  v15 = *(unsigned __int8 *)(a1 + 8);
  *(_QWORD *)(a1 + 40) = v12;
  result = ((_DWORD)a8 << 8) | v15;
  *(_DWORD *)(a1 + 8) = result;
  if ( 4 * a8 )
    return (unsigned __int64)memcpy(v12, src, 4 * a8);
  return result;
}

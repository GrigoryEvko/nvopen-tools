// Function: sub_A52810
// Address: 0xa52810
//
__int64 __fastcall sub_A52810(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // rax
  unsigned int v5; // ebx
  void *v6; // rdx
  __int64 v7; // rdi
  __int64 result; // rax

  v3 = a3;
  if ( !a1 )
    return sub_904010(a3, " <cannot get addrspace!>");
  v4 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  v5 = *(_DWORD *)(v4 + 8) >> 8;
  if ( v5 || (result = sub_A4F760(a2)) == 0 || *(_DWORD *)(result + 320) )
  {
    v6 = *(void **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v6 <= 0xAu )
    {
      v3 = sub_CB6200(v3, " addrspace(", 11);
    }
    else
    {
      qmemcpy(v6, " addrspace(", 11);
      *(_QWORD *)(v3 + 32) += 11LL;
    }
    v7 = sub_CB59D0(v3, v5);
    result = *(_QWORD *)(v7 + 32);
    if ( *(_QWORD *)(v7 + 24) == result )
    {
      return sub_CB6200(v7, ")", 1);
    }
    else
    {
      *(_BYTE *)result = 41;
      ++*(_QWORD *)(v7 + 32);
    }
  }
  return result;
}

// Function: sub_B4E7F0
// Address: 0xb4e7f0
//
__int64 __fastcall sub_B4E7F0(__int64 a1, void *a2, __int64 a3)
{
  __int64 v3; // rax
  size_t v4; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 result; // rax

  LODWORD(v3) = 0;
  v4 = 4 * a3;
  v6 = (4 * a3) >> 2;
  v8 = *(unsigned int *)(a1 + 84);
  *(_DWORD *)(a1 + 80) = 0;
  v9 = 0;
  if ( v6 > v8 )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v6, 4);
    v3 = *(unsigned int *)(a1 + 80);
    v9 = 4 * v3;
  }
  if ( v4 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 72) + v9), a2, v4);
    LODWORD(v3) = *(_DWORD *)(a1 + 80);
  }
  v10 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 80) = v3 + v6;
  result = sub_B4E660((int *)a2, a3, v10);
  *(_QWORD *)(a1 + 104) = result;
  return result;
}

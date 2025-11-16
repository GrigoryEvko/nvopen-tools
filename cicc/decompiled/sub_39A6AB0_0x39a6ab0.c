// Function: sub_39A6AB0
// Address: 0x39a6ab0
//
__int64 __fastcall sub_39A6AB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  void *v11; // rcx
  size_t v12; // rdx
  size_t v13; // r8

  v5 = sub_39A5A90((__int64)a1, 47, a2, 0);
  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_QWORD *)(a3 + 8 * (1 - v6));
  if ( v7 )
  {
    sub_39A6760(a1, v5, v7, 73);
    v6 = *(unsigned int *)(a3 + 8);
  }
  result = -v6;
  v9 = *(_QWORD *)(a3 + 8 * result);
  if ( v9 )
  {
    result = sub_161E970(v9);
    if ( v10 )
    {
      v11 = *(void **)(a3 - 8LL * *(unsigned int *)(a3 + 8));
      if ( v11 )
      {
        v11 = (void *)sub_161E970(*(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8)));
        v13 = v12;
      }
      else
      {
        v13 = 0;
      }
      return sub_39A3F30(a1, v5, 3, v11, v13);
    }
  }
  return result;
}

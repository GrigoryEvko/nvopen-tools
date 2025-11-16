// Function: sub_894B30
// Address: 0x894b30
//
__int64 *__fastcall sub_894B30(__int64 a1, __int64 a2, const __m128i *a3, int a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-50h] BYREF
  int v18; // [rsp+10h] [rbp-40h]

  v17[0] = a1;
  v9 = *(unsigned __int8 **)(a2 + 376);
  v17[1] = a3;
  v18 = a4;
  if ( !v9 )
  {
    v15 = sub_881A70(0, 0xBu, 12, 33, a5, a6);
    *(_QWORD *)(a2 + 376) = v15;
    v9 = (unsigned __int8 *)v15;
  }
  v10 = (__int64 *)sub_881B20(v9, (__int64)v17, a5 != 0);
  v12 = v10;
  if ( v10 )
  {
    v13 = *v10;
    if ( !v13 )
    {
      v16 = sub_878410(v9, v17, v11);
      *v12 = v16;
      *(_QWORD *)(v16 + 8) = sub_72F240(a3);
      *(_QWORD *)(*v12 + 24) = a5;
      *(_DWORD *)(*v12 + 16) = a4;
      v13 = *v12;
    }
    return *(__int64 **)(v13 + 24);
  }
  return v12;
}

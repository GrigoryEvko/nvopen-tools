// Function: sub_3946CB0
// Address: 0x3946cb0
//
__int64 __fastcall sub_3946CB0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        size_t a4,
        unsigned __int8 *a5,
        size_t a6,
        unsigned __int8 *a7,
        size_t a8)
{
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r14
  int v13; // eax
  _QWORD *v14; // rsi
  __int64 v16; // [rsp+8h] [rbp-38h]

  v10 = sub_16D1B30((__int64 *)a2, a3, a4);
  if ( v10 == -1 )
    return 0;
  v11 = *(_QWORD *)a2 + 8LL * v10;
  if ( v11 == *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) )
    return 0;
  v12 = *(_QWORD *)v11;
  v16 = *(_QWORD *)a2 + 8LL * v10;
  v13 = sub_16D1B30((__int64 *)(*(_QWORD *)v11 + 8LL), a7, a8);
  v14 = (_QWORD *)(v13 == -1
                 ? *(_QWORD *)(v12 + 8) + 8LL * *(unsigned int *)(v12 + 16)
                 : *(_QWORD *)(v12 + 8) + 8LL * v13);
  if ( v14 == (_QWORD *)(*(_QWORD *)(*(_QWORD *)v16 + 8LL) + 8LL * *(unsigned int *)(*(_QWORD *)v16 + 16LL)) )
    return 0;
  else
    return sub_3946C00(*v14 + 8LL, a5, a6);
}

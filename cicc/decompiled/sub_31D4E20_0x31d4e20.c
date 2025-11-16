// Function: sub_31D4E20
// Address: 0x31d4e20
//
__int64 __fastcall sub_31D4E20(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r14
  __int64 v5; // rdi
  __int64 *v6; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v9; // rdi

  if ( *(_BYTE *)(a2 + 261) )
  {
    v3 = *(__int64 **)(a1 + 576);
    v4 = &v3[*(unsigned int *)(a1 + 584)];
    while ( v4 != v3 )
    {
      v5 = *v3++;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 64LL))(v5, a2);
    }
    v6 = *(__int64 **)(a1 + 552);
    result = *(unsigned int *)(a1 + 560);
    for ( i = &v6[result]; i != v6; result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 64LL))(
                                               v9,
                                               a2) )
      v9 = *v6++;
  }
  return result;
}

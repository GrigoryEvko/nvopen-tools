// Function: sub_B6E9A0
// Address: 0xb6e9a0
//
__int64 __fastcall sub_B6E9A0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rdi

  result = *a1;
  v3 = *a2;
  *a2 = 0;
  v4 = *(_QWORD *)(result + 96);
  *(_QWORD *)(result + 96) = v3;
  if ( v4 )
  {
    if ( *(_BYTE *)(v4 + 64) )
    {
      v6 = *(_QWORD *)(v4 + 32);
      *(_BYTE *)(v4 + 64) = 0;
      if ( v6 != v4 + 48 )
        j_j___libc_free_0(v6, *(_QWORD *)(v4 + 48) + 1LL);
    }
    v5 = *(_QWORD *)(v4 + 24);
    if ( v5 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
    if ( *(_BYTE *)(v4 + 16) )
    {
      *(_BYTE *)(v4 + 16) = 0;
      sub_C88FF0((void *)v4);
    }
    return j_j___libc_free_0(v4, 72);
  }
  return result;
}

// Function: sub_104C0D0
// Address: 0x104c0d0
//
__int64 __fastcall sub_104C0D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a1 + 224);
  result = *(unsigned int *)(a1 + 232);
  v5 = v3 + 8 * result;
  while ( v3 != v5 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v5 - 8);
      v5 -= 8;
      if ( !v6 )
        break;
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 != v6 + 40 )
        _libc_free(v7, a2);
      a2 = 80;
      result = j_j___libc_free_0(v6, 80);
      if ( v3 == v5 )
        goto LABEL_7;
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_BYTE *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 316) = 0;
  return result;
}

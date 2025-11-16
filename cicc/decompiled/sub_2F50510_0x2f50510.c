// Function: sub_2F50510
// Address: 0x2f50510
//
__int64 __fastcall sub_2F50510(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 872);
  *(_QWORD *)(a1 + 872) = 0;
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2);
  result = *(unsigned int *)(a1 + 24184);
  v4 = *(_QWORD **)(a1 + 24176);
  v5 = &v4[18 * result];
  while ( v4 != v5 )
  {
    while ( 1 )
    {
      v5 -= 18;
      v6 = v5[12];
      if ( (_QWORD *)v6 != v5 + 14 )
        _libc_free(v6);
      v7 = v5[3];
      if ( (_QWORD *)v7 != v5 + 5 )
        _libc_free(v7);
      result = v5[1];
      v5[2] = 0;
      if ( !result )
        break;
      --*(_DWORD *)(result + 8);
      if ( v4 == v5 )
        goto LABEL_11;
    }
  }
LABEL_11:
  *(_DWORD *)(a1 + 24184) = 0;
  return result;
}

// Function: sub_1EBBCC0
// Address: 0x1ebbcc0
//
__int64 __fastcall sub_1EBBCC0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rdi

  v2 = *(_QWORD *)(a1 + 872);
  *(_QWORD *)(a1 + 872) = 0;
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 16LL))(v2);
  result = *(unsigned int *)(a1 + 24176);
  v4 = *(_QWORD **)(a1 + 24168);
  *(_DWORD *)(a1 + 928) = 0;
  v5 = &v4[12 * result];
  while ( v4 != v5 )
  {
    while ( 1 )
    {
      v5 -= 12;
      v6 = v5[6];
      if ( (_QWORD *)v6 != v5 + 8 )
        _libc_free(v6);
      _libc_free(v5[3]);
      result = v5[1];
      v5[2] = 0;
      if ( !result )
        break;
      --*(_DWORD *)(result + 8);
      if ( v4 == v5 )
        goto LABEL_9;
    }
  }
LABEL_9:
  *(_DWORD *)(a1 + 24176) = 0;
  return result;
}

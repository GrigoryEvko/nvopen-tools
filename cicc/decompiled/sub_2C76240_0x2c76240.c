// Function: sub_2C76240
// Address: 0x2c76240
//
struct __jmp_buf_tag *__fastcall sub_2C76240(__int64 a1, __int64 a2, __int64 a3, char *a4)
{
  _BYTE *v5; // rax
  struct __jmp_buf_tag *result; // rax
  __int64 v7; // rdi

  v5 = *(_BYTE **)(a1 + 16);
  if ( v5 )
    *v5 = 0;
  result = (struct __jmp_buf_tag *)*(unsigned int *)(a1 + 4);
  if ( !(_DWORD)result )
  {
    v7 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(v7 + 32) != *(_QWORD *)(v7 + 16) )
    {
      sub_CB5AE0((__int64 *)v7);
      v7 = *(_QWORD *)(a1 + 24);
    }
    return sub_CEB520(*(_QWORD **)(v7 + 48), a2, a3, a4);
  }
  return result;
}

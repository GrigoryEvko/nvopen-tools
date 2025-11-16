// Function: sub_C8CE00
// Address: 0xc8ce00
//
__int64 __fastcall sub_C8CE00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9

  v7 = *(_BYTE *)(a3 + 28) == 0;
  v8 = *(unsigned __int8 *)(a1 + 28);
  if ( v7 )
  {
    v10 = *(unsigned int *)(a3 + 16);
    if ( *(_DWORD *)(a1 + 16) == (_DWORD)v10 )
      return sub_C8CD20(a1, a3);
    v11 = 8 * v10;
    if ( (_BYTE)v8 )
    {
      v12 = malloc(8 * v10, a2, v8, a4, a5, a6);
      if ( v12 )
        goto LABEL_9;
    }
    else
    {
      a2 = 8 * v10;
      v12 = realloc(*(void **)(a1 + 8));
      if ( v12 )
      {
LABEL_9:
        *(_QWORD *)(a1 + 8) = v12;
        *(_BYTE *)(a1 + 28) = 0;
        return sub_C8CD20(a1, a3);
      }
    }
    if ( v11 || (v12 = malloc(1, a2, v13, v14, v15, v16)) == 0 )
      sub_C64F00("Allocation failed", 1u);
    goto LABEL_9;
  }
  if ( !(_BYTE)v8 )
    _libc_free(*(_QWORD *)(a1 + 8), a2);
  *(_QWORD *)(a1 + 8) = a2;
  *(_BYTE *)(a1 + 28) = 1;
  return sub_C8CD20(a1, a3);
}

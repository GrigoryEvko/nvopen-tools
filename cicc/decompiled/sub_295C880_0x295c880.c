// Function: sub_295C880
// Address: 0x295c880
//
void __fastcall sub_295C880(unsigned __int64 *a1, _QWORD *a2, __int64 a3)
{
  signed __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rbx
  void *v9; // rdi
  __int64 v10; // r15

  if ( !a3 )
  {
    *a1 = 0;
    return;
  }
  if ( a3 == 1 )
  {
    *a1 = *a2 & 0xFFFFFFFFFFFFFFFBLL;
    return;
  }
  v4 = 8 * a3;
  v5 = (_QWORD *)sub_22077B0(0x30u);
  v8 = (unsigned __int64)v5;
  if ( v5 )
  {
    v9 = v5 + 2;
    *v5 = v5 + 2;
    v10 = v4 >> 3;
    v5[1] = 0x400000000LL;
    if ( (unsigned __int64)v4 > 0x20 )
    {
      sub_C8D5F0((__int64)v5, v9, v4 >> 3, 8u, v6, v7);
      v9 = (void *)(*(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8));
    }
    else if ( !v4 )
    {
LABEL_8:
      *(_DWORD *)(v8 + 8) = v10 + v4;
      goto LABEL_9;
    }
    memcpy(v9, a2, v4);
    LODWORD(v4) = *(_DWORD *)(v8 + 8);
    goto LABEL_8;
  }
LABEL_9:
  *a1 = v8 | 4;
}

// Function: sub_23AF0B0
// Address: 0x23af0b0
//
__int64 __fastcall sub_23AF0B0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rax
  _QWORD *v10; // rdi

  *(_QWORD *)a1 = &unk_4A161A0;
  v2 = *(unsigned int *)(a1 + 72);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 56);
    v4 = v3 + 40 * v2;
    do
    {
      if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v3 + 16), 16LL * *(unsigned int *)(v3 + 32), 8);
      v3 += 40;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 72);
  }
  result = sub_C7D6A0(*(_QWORD *)(a1 + 56), 40 * v2, 8);
  if ( *(_BYTE *)(a1 + 40) )
  {
    v6 = *(unsigned int *)(a1 + 32);
    *(_BYTE *)(a1 + 40) = 0;
    if ( (_DWORD)v6 )
    {
      v7 = *(__int64 **)(a1 + 16);
      v8 = &v7[5 * v6];
      do
      {
        while ( 1 )
        {
          if ( *v7 <= 0x7FFFFFFFFFFFFFFDLL )
          {
            v7[1] = (__int64)&unk_49DB368;
            v9 = v7[4];
            if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
              break;
          }
          v7 += 5;
          if ( v8 == v7 )
            goto LABEL_16;
        }
        v10 = v7 + 2;
        v7 += 5;
        sub_BD60C0(v10);
      }
      while ( v8 != v7 );
LABEL_16:
      v6 = *(unsigned int *)(a1 + 32);
    }
    return sub_C7D6A0(*(_QWORD *)(a1 + 16), 40 * v6, 8);
  }
  return result;
}

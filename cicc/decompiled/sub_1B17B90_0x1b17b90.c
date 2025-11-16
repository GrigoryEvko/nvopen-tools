// Function: sub_1B17B90
// Address: 0x1b17b90
//
__int64 __fastcall sub_1B17B90(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r15
  _BYTE *v11; // rdi
  const void *v12; // rax
  __int64 v13; // rdx
  bool v14; // zf

  v5 = sub_13FD000(a2);
  if ( v5 && (v6 = v5, v7 = *(unsigned int *)(v5 + 8), (unsigned int)v7 > 1) )
  {
    v8 = v7;
    v9 = 1;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v6 + 8 * (v9 - v7));
      if ( (unsigned __int8)(*(_BYTE *)v10 - 4) <= 0x1Eu )
      {
        v11 = *(_BYTE **)(v10 - 8LL * *(unsigned int *)(v10 + 8));
        if ( !*v11 )
        {
          v12 = (const void *)sub_161E970((__int64)v11);
          if ( v13 == a4 && (!a4 || !memcmp(a3, v12, a4)) )
            break;
        }
      }
      if ( v8 == ++v9 )
        goto LABEL_13;
      v7 = *(unsigned int *)(v6 + 8);
    }
    v14 = *(_DWORD *)(v10 + 8) == 1;
    *(_BYTE *)(a1 + 8) = 1;
    if ( v14 )
      *(_QWORD *)a1 = 0;
    else
      *(_QWORD *)a1 = v10 - 8;
  }
  else
  {
LABEL_13:
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a1;
}

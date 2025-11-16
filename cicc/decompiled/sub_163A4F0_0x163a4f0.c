// Function: sub_163A4F0
// Address: 0x163a4f0
//
__int64 __fastcall sub_163A4F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax
  __int64 result; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r12
  _QWORD *v16; // rbx
  void (*v17)(); // rax

  v6 = a2;
  v7 = a1;
  if ( (unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
  {
    sub_16C9040(a1);
    if ( !*(_DWORD *)(a1 + 32) )
      goto LABEL_3;
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 32);
    ++*(_DWORD *)(a1 + 8);
    if ( !v12 )
      goto LABEL_3;
  }
  v14 = *(_QWORD **)(a1 + 24);
  v15 = &v14[2 * *(unsigned int *)(a1 + 40)];
  if ( v14 != v15 )
  {
    while ( 1 )
    {
      v8 = *v14;
      v16 = v14;
      if ( *v14 != -8 && v8 != -4 )
        break;
      v14 += 2;
      if ( v15 == v14 )
        goto LABEL_3;
    }
    if ( v14 != v15 )
    {
      v17 = *(void (**)())(*(_QWORD *)a2 + 24LL);
      if ( v17 != nullsub_574 )
        goto LABEL_20;
      while ( 1 )
      {
        v16 += 2;
        if ( v16 == v15 )
          break;
        while ( *v16 == -8 || *v16 == -4 )
        {
          v16 += 2;
          if ( v15 == v16 )
            goto LABEL_3;
        }
        if ( v16 == v15 )
          break;
        v17 = *(void (**)())(*(_QWORD *)v6 + 24LL);
        if ( v17 != nullsub_574 )
        {
LABEL_20:
          a2 = v16[1];
          a1 = v6;
          ((void (__fastcall *)(__int64, __int64))v17)(v6, a2);
        }
      }
    }
  }
LABEL_3:
  result = sub_16D5D40(a1, a2, v8, v9, v10, v11);
  if ( (_BYTE)result )
    return sub_16C9060(v7);
  --*(_DWORD *)(v7 + 8);
  return result;
}

// Function: sub_D13750
// Address: 0xd13750
//
__int64 __fastcall sub_D13750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  char v13; // dl
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // r14
  __int64 v17; // rax

  if ( a2 )
  {
    v7 = a2;
    while ( 1 )
    {
      v8 = *(_QWORD *)a1;
      v9 = *(unsigned int *)(*(_QWORD *)a1 + 20LL);
      v10 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 20LL) - *(_DWORD *)(*(_QWORD *)a1 + 24LL));
      if ( (unsigned int)v10 >= **(_DWORD **)(a1 + 8) )
      {
        (*(void (__fastcall **)(_QWORD))(***(_QWORD ***)(a1 + 16) + 16LL))(**(_QWORD **)(a1 + 16));
        return 0;
      }
      if ( !*(_BYTE *)(v8 + 28) )
        goto LABEL_11;
      v11 = *(_QWORD **)(v8 + 8);
      v10 = (__int64)&v11[(unsigned int)v9];
      if ( v11 != (_QWORD *)v10 )
      {
        while ( *v11 != v7 )
        {
          if ( (_QWORD *)v10 == ++v11 )
            goto LABEL_17;
        }
        goto LABEL_9;
      }
LABEL_17:
      if ( (unsigned int)v9 < *(_DWORD *)(v8 + 16) )
      {
        *(_DWORD *)(v8 + 20) = v9 + 1;
        *(_QWORD *)v10 = v7;
        ++*(_QWORD *)v8;
LABEL_12:
        v14 = **(_QWORD **)(a1 + 16);
        v15 = *(__int64 (**)())(*(_QWORD *)v14 + 24LL);
        if ( v15 == sub_D13590 || ((unsigned __int8 (__fastcall *)(__int64, __int64))v15)(v14, v7) )
        {
          v16 = *(_QWORD *)(a1 + 24);
          v17 = *(unsigned int *)(v16 + 8);
          if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 12) )
          {
            sub_C8D5F0(*(_QWORD *)(a1 + 24), (const void *)(v16 + 16), v17 + 1, 8u, a5, a6);
            v17 = *(unsigned int *)(v16 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v16 + 8 * v17) = v7;
          ++*(_DWORD *)(v16 + 8);
          v7 = *(_QWORD *)(v7 + 8);
          if ( !v7 )
            return 1;
        }
        else
        {
          v7 = *(_QWORD *)(v7 + 8);
          if ( !v7 )
            return 1;
        }
      }
      else
      {
LABEL_11:
        sub_C8CC70(v8, v7, v10, v9, a5, a6);
        if ( v13 )
          goto LABEL_12;
LABEL_9:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return 1;
      }
    }
  }
  return 1;
}

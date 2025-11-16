// Function: sub_30B0D80
// Address: 0x30b0d80
//
void __fastcall sub_30B0D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // rdx
  const void *v12; // rsi
  __int64 v13; // rdi

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_4A32450;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_DWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 3;
  *(_QWORD *)(a1 + 72) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 8) )
  {
    v13 = a1 + 64;
    if ( v13 == a2 )
    {
      nullsub_2027();
    }
    else
    {
      v7 = *(unsigned int *)(a2 + 8);
      v8 = *(unsigned int *)(v13 + 8);
      v9 = *(_DWORD *)(a2 + 8);
      if ( v7 > v8 )
      {
        if ( v7 > *(unsigned int *)(v13 + 12) )
        {
          *(_DWORD *)(v13 + 8) = 0;
          v10 = 0;
          sub_C8D5F0(v13, (const void *)(v13 + 16), v7, 8u, a5, a6);
          v7 = *(unsigned int *)(a2 + 8);
        }
        else
        {
          v10 = 8 * v8;
          if ( *(_DWORD *)(v13 + 8) )
          {
            memmove(*(void **)v13, *(const void **)a2, 8 * v8);
            v7 = *(unsigned int *)(a2 + 8);
          }
        }
        v11 = 8 * v7;
        v12 = (const void *)(*(_QWORD *)a2 + v10);
        if ( v12 != (const void *)(v11 + *(_QWORD *)a2) )
          memcpy((void *)(v10 + *(_QWORD *)v13), v12, v11 - v10);
        goto LABEL_9;
      }
      if ( !*(_DWORD *)(a2 + 8) )
      {
LABEL_9:
        *(_DWORD *)(v13 + 8) = v9;
        return;
      }
      memmove(*(void **)v13, *(const void **)a2, 8 * v7);
      *(_DWORD *)(v13 + 8) = v9;
    }
  }
}

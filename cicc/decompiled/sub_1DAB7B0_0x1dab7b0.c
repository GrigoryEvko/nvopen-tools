// Function: sub_1DAB7B0
// Address: 0x1dab7b0
//
char __fastcall sub_1DAB7B0(__int64 a1, __int64 a2)
{
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // r13
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rdi
  int *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rdx

  v7 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v5 = *(_QWORD *)v7;
  v6 = *(unsigned int *)(v7 + 12);
  LODWORD(v7) = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  v8 = (__int64 *)(v5 + 16 * v6);
  v9 = *(_DWORD *)(*(_QWORD *)a1 + 80LL) == 0;
  v10 = *v8;
  v11 = *v8;
  if ( v9 )
  {
    if ( (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10 >> 1) & 3) <= (unsigned int)v7 )
      goto LABEL_3;
    v12 = (int *)(v5 + 4 * v6 + 64);
LABEL_5:
    LOBYTE(v7) = sub_1DA98F0(a1, a2, *v12);
    if ( !(_BYTE)v7 )
      goto LABEL_3;
    v14 = *(_QWORD *)(a1 + 8);
    v15 = *(unsigned int *)(a1 + 16);
    v16 = v14 + 16 * v15 - 16;
    v17 = *(_DWORD *)(v16 + 12);
    if ( v17 )
    {
      if ( (_DWORD)v15 && (v15 = *(unsigned int *)(v14 + 8), *(_DWORD *)(v14 + 12) < (unsigned int)v15)
        || (v13 = *(unsigned int *)(*(_QWORD *)a1 + 80LL), !(_DWORD)v13) )
      {
        *(_DWORD *)(v16 + 12) = v17 - 1;
LABEL_15:
        v18 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
        v19 = *(_QWORD *)(*(_QWORD *)v18 + 16LL * *(unsigned int *)(v18 + 12));
        sub_1DAB460(a1, v16, v18, v15, v13);
        v20 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
        v7 = *(_QWORD *)v20 + 16LL * *(unsigned int *)(v20 + 12);
        *(_QWORD *)v7 = v19;
        return v7;
      }
    }
    else
    {
      LODWORD(v13) = *(_DWORD *)(*(_QWORD *)a1 + 80LL);
    }
    v16 = (unsigned int)v13;
    sub_3945E40(a1 + 8, (unsigned int)v13);
    goto LABEL_15;
  }
  if ( (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10 >> 1) & 3) > (unsigned int)v7 )
  {
    v12 = (int *)(v5 + 4 * v6 + 144);
    goto LABEL_5;
  }
LABEL_3:
  *v8 = a2;
  return v7;
}

// Function: sub_D695C0
// Address: 0xd695c0
//
__int64 __fastcall sub_D695C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 *v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // rdx
  char v15; // dl
  char v16; // dl
  int v17; // eax

  v8 = (__int64)a3;
  v9 = *(unsigned __int8 *)(a2 + 28);
  if ( !(_BYTE)v9 )
    goto LABEL_10;
  v10 = *(__int64 **)(a2 + 8);
  a3 = &v10[*(unsigned int *)(a2 + 20)];
  a5 = *(unsigned int *)(a2 + 20);
  if ( v10 == a3 )
  {
LABEL_9:
    if ( (unsigned int)a5 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = a5 + 1;
      *a3 = v8;
      v10 = *(__int64 **)(a2 + 8);
      v16 = *(_BYTE *)(a2 + 28);
      v12 = *(_QWORD *)a2 + 1LL;
      v17 = *(_DWORD *)(a2 + 20);
      *(_QWORD *)a2 = v12;
      v11 = &v10[v17 - 1];
LABEL_11:
      if ( !v16 )
      {
        v13 = &v10[*(unsigned int *)(a2 + 16)];
        goto LABEL_15;
      }
      goto LABEL_7;
    }
LABEL_10:
    v11 = sub_C8CC70(a2, v8, (__int64)a3, v9, a5, a6);
    v12 = *(_QWORD *)a2;
    v10 = *(__int64 **)(a2 + 8);
    LOBYTE(v9) = v15;
    v16 = *(_BYTE *)(a2 + 28);
    goto LABEL_11;
  }
  v11 = *(__int64 **)(a2 + 8);
  while ( v8 != *v11 )
  {
    if ( a3 == ++v11 )
      goto LABEL_9;
  }
  v12 = *(_QWORD *)a2;
  LOBYTE(v9) = 0;
LABEL_7:
  v13 = &v10[*(unsigned int *)(a2 + 20)];
  while ( v11 != v13 )
  {
    if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v11;
LABEL_15:
    ;
  }
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = v13;
  *(_QWORD *)(a1 + 24) = v12;
  *(_BYTE *)(a1 + 32) = v9;
  return a1;
}

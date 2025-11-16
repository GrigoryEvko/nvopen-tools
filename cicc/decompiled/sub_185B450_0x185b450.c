// Function: sub_185B450
// Address: 0x185b450
//
void __fastcall sub_185B450(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 *v6; // rax
  __int64 *v7; // r13
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  char v11; // dl
  __int128 v12; // rdi
  __int64 v13; // rax
  int v14; // r14d
  unsigned int i; // r15d
  __int64 v16; // rbx
  __int64 v17; // r8
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp-40h] [rbp-40h]

  if ( a3 )
  {
    v6 = *(__int64 **)a1;
    v7 = *(__int64 **)(a2 - 24);
    v8 = **(_QWORD **)a1;
    v9 = *v7;
    if ( !v8 )
      goto LABEL_6;
    *(_QWORD *)&v12 = **(_QWORD **)(v8 - 24);
    v10 = *(_QWORD *)(a1 + 8);
    v11 = *(_BYTE *)(v12 + 8);
    *((_QWORD *)&v12 + 1) = *(_QWORD *)v10;
    if ( v11 == 13 )
    {
      v13 = sub_159F090((__int64 **)v12, *((__int64 **)&v12 + 1), *(unsigned int *)(v10 + 8), a4);
      sub_15E5440(v8, v13);
    }
    else
    {
      v20 = v11 == 14
          ? sub_159DFD0(v12, *(unsigned int *)(v10 + 8), a4)
          : sub_15A01B0(*(__int64 **)v10, *(unsigned int *)(v10 + 8));
      sub_15E5440(v8, v20);
    }
    v6 = *(__int64 **)a1;
    if ( **(_QWORD **)a1 != a2 )
    {
LABEL_6:
      *v6 = a2;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = 0;
      if ( *(_BYTE *)(v9 + 8) == 13 )
        v14 = *(_DWORD *)(v9 + 12);
      else
        v14 = *(_DWORD *)(v9 + 32);
      if ( v14 )
      {
        for ( i = 0; i != v14; ++i )
        {
          v16 = *(_QWORD *)(a1 + 8);
          v17 = sub_15A0A60((__int64)v7, i);
          v19 = *(unsigned int *)(v16 + 8);
          if ( (unsigned int)v19 >= *(_DWORD *)(v16 + 12) )
          {
            v21 = v17;
            sub_16CD150(v16, (const void *)(v16 + 16), 0, 8, v17, v18);
            v19 = *(unsigned int *)(v16 + 8);
            v17 = v21;
          }
          *(_QWORD *)(*(_QWORD *)v16 + 8 * v19) = v17;
          ++*(_DWORD *)(v16 + 8);
        }
      }
    }
  }
}

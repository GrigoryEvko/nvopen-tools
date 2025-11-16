// Function: sub_3089D70
// Address: 0x3089d70
//
void __fastcall sub_3089D70(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // edx
  int *v6; // r13
  int v7; // esi
  unsigned __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // edx
  int *v12; // r13
  int v13; // ecx
  unsigned __int64 v14; // r12
  int v15; // r8d
  int v16; // r8d

  v3 = *(unsigned int *)(a1 + 224);
  v4 = *(_QWORD *)(a1 + 208);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (37 * a2);
    v6 = (int *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
LABEL_3:
      if ( v6 != (int *)(v4 + 16 * v3) )
      {
        v8 = (unsigned __int64 *)*((_QWORD *)v6 + 1);
        if ( v8 )
        {
          if ( (unsigned __int64 *)*v8 != v8 + 2 )
            _libc_free(*v8);
          j_j___libc_free_0((unsigned __int64)v8);
        }
        *v6 = 0x80000000;
        --*(_DWORD *)(a1 + 216);
        ++*(_DWORD *)(a1 + 220);
      }
    }
    else
    {
      v15 = 1;
      while ( v7 != 0x7FFFFFFF )
      {
        v5 = (v3 - 1) & (v15 + v5);
        v6 = (int *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_3;
        ++v15;
      }
    }
  }
  v9 = *(unsigned int *)(a1 + 352);
  v10 = *(_QWORD *)(a1 + 336);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (37 * a2);
    v12 = (int *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == a2 )
    {
LABEL_11:
      if ( v12 != (int *)(v10 + 16 * v9) )
      {
        v14 = *((_QWORD *)v12 + 1);
        if ( v14 )
        {
          sub_C7D6A0(*(_QWORD *)(v14 + 8), 16LL * *(unsigned int *)(v14 + 24), 8);
          j_j___libc_free_0(v14);
        }
        *v12 = 0x80000000;
        --*(_DWORD *)(a1 + 344);
        ++*(_DWORD *)(a1 + 348);
      }
    }
    else
    {
      v16 = 1;
      while ( v13 != 0x7FFFFFFF )
      {
        v11 = (v9 - 1) & (v16 + v11);
        v12 = (int *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == a2 )
          goto LABEL_11;
        ++v16;
      }
    }
  }
}

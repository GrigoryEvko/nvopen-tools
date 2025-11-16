// Function: sub_1EE72E0
// Address: 0x1ee72e0
//
void __fastcall sub_1EE72E0(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // rbx
  unsigned int *v5; // r12
  unsigned int v6; // ecx
  __int64 v7; // r9
  unsigned int v8; // esi
  unsigned __int64 v9; // r14
  _BYTE *v10; // r8
  unsigned int v11; // eax
  __int64 v12; // rdi
  _DWORD *v13; // rdx
  int v14; // r8d
  unsigned int v15; // ecx
  int v16; // esi
  __int64 v17; // rax

  v3 = &a2[2 * a3];
  if ( a2 != v3 )
  {
    v5 = a2;
    do
    {
      v6 = *v5;
      v7 = v5[1];
      if ( (*v5 & 0x80000000) != 0 )
        v6 = *(_DWORD *)(a1 + 192) + (v6 & 0x7FFFFFFF);
      v8 = *(_DWORD *)(a1 + 104);
      v9 = v6 | (unsigned __int64)(v7 << 32);
      v10 = (_BYTE *)(*(_QWORD *)(a1 + 176) + v6);
      v11 = (unsigned __int8)*v10;
      if ( v11 >= v8 )
        goto LABEL_13;
      v12 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v13 = (_DWORD *)(v12 + 8LL * v11);
        if ( v6 == *v13 )
          break;
        v11 += 256;
        if ( v8 <= v11 )
          goto LABEL_13;
      }
      if ( v13 == (_DWORD *)(v12 + 8LL * v8) )
      {
LABEL_13:
        *v10 = v8;
        v17 = *(unsigned int *)(a1 + 104);
        if ( (unsigned int)v17 >= *(_DWORD *)(a1 + 108) )
        {
          sub_16CD150(a1 + 96, (const void *)(a1 + 112), 0, 8, (int)v10, v7);
          v17 = *(unsigned int *)(a1 + 104);
        }
        v14 = 0;
        *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v17) = v9;
        ++*(_DWORD *)(a1 + 104);
      }
      else
      {
        v14 = v13[1];
        v13[1] = v14 | v7;
      }
      v15 = v5[1];
      v16 = *v5;
      v5 += 2;
      sub_1EE5D10(a1, v16, v14, v14 | v15);
    }
    while ( v3 != v5 );
  }
}

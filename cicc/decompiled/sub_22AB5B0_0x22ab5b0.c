// Function: sub_22AB5B0
// Address: 0x22ab5b0
//
void __fastcall sub_22AB5B0(unsigned __int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  _QWORD *v4; // rdi
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rcx
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rdx
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rax

  v2 = *(_QWORD *)(a1 + 48);
  v3 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v2 + 68) )
  {
    v4 = *(_QWORD **)(v2 + 48);
    v5 = &v4[*(unsigned int *)(v2 + 60)];
    v6 = v4;
    if ( v4 != v5 )
    {
      while ( v3 != *v6 )
      {
        if ( v5 == ++v6 )
          goto LABEL_7;
      }
      v7 = (unsigned int)(*(_DWORD *)(v2 + 60) - 1);
      *(_DWORD *)(v2 + 60) = v7;
      *v6 = v4[v7];
      ++*(_QWORD *)(v2 + 40);
    }
  }
  else
  {
    v13 = sub_C8CA60(v2 + 40, v3);
    if ( v13 )
    {
      *v13 = -2;
      ++*(_DWORD *)(v2 + 64);
      ++*(_QWORD *)(v2 + 40);
    }
  }
LABEL_7:
  v8 = *(unsigned __int64 **)(a1 + 40);
  v9 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
  *v8 = v9 | *v8 & 7;
  *(_QWORD *)(v9 + 8) = v8;
  *(_QWORD *)(a1 + 32) &= 7uLL;
  v10 = *(_BYTE *)(a1 + 108) == 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)a1 = &unk_4A09CC0;
  if ( v10 )
    _libc_free(*(_QWORD *)(a1 + 88));
  v11 = *(_QWORD *)(a1 + 72);
  if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
    sub_BD60C0((_QWORD *)(a1 + 56));
  *(_QWORD *)a1 = &unk_49DB368;
  v12 = *(_QWORD *)(a1 + 24);
  if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
    sub_BD60C0((_QWORD *)(a1 + 8));
  j_j___libc_free_0(a1);
}

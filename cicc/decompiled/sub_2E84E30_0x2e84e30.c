// Function: sub_2E84E30
// Address: 0x2e84e30
//
void __fastcall sub_2E84E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  void **v9; // rax
  void **v10; // rsi
  __int64 v11; // rdi
  void **v12; // rdx

  v8 = *(unsigned int *)(a2 + 152);
  v9 = *(void ***)(a2 + 144);
  *(_BYTE *)(a2 + 160) = 1;
  v10 = &v9[v8];
  v11 = (8 * v8) >> 3;
  if ( (8 * v8) >> 5 )
  {
    v12 = &v9[4 * ((8 * v8) >> 5)];
    while ( *v9 != &unk_5025C1C )
    {
      if ( v9[1] == &unk_5025C1C )
      {
        ++v9;
        break;
      }
      if ( v9[2] == &unk_5025C1C )
      {
        v9 += 2;
        break;
      }
      if ( v9[3] == &unk_5025C1C )
      {
        v9 += 3;
        break;
      }
      v9 += 4;
      if ( v9 == v12 )
      {
        v11 = v10 - v9;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v10 != v9 )
      goto LABEL_9;
    goto LABEL_15;
  }
LABEL_11:
  if ( v11 != 2 )
  {
    if ( v11 != 3 )
    {
      if ( v11 != 1 )
        goto LABEL_15;
      goto LABEL_14;
    }
    if ( *v9 == &unk_5025C1C )
      goto LABEL_8;
    ++v9;
  }
  if ( *v9 == &unk_5025C1C )
    goto LABEL_8;
  ++v9;
LABEL_14:
  if ( *v9 == &unk_5025C1C )
    goto LABEL_8;
LABEL_15:
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
  {
    sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v8 + 1, 8u, a5, a6);
    v10 = (void **)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
  }
  *v10 = &unk_5025C1C;
  ++*(_DWORD *)(a2 + 152);
LABEL_9:
  sub_2E84680(a1, a2);
}

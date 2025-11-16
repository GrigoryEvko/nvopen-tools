// Function: sub_2DDB5A0
// Address: 0x2ddb5a0
//
void __fastcall sub_2DDB5A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  void **v8; // rax
  void **v9; // rsi
  __int64 v10; // rdi
  void **v11; // rdx

  v7 = *(unsigned int *)(a2 + 152);
  v8 = *(void ***)(a2 + 144);
  v9 = &v8[v7];
  v10 = (8 * v7) >> 3;
  if ( (8 * v7) >> 5 )
  {
    v11 = &v8[4 * ((8 * v7) >> 5)];
    while ( *v8 != &unk_4F8780C )
    {
      if ( v8[1] == &unk_4F8780C )
      {
        ++v8;
        break;
      }
      if ( v8[2] == &unk_4F8780C )
      {
        v8 += 2;
        break;
      }
      if ( v8[3] == &unk_4F8780C )
      {
        v8 += 3;
        break;
      }
      v8 += 4;
      if ( v11 == v8 )
      {
        v10 = v9 - v8;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v9 != v8 )
      goto LABEL_9;
    goto LABEL_15;
  }
LABEL_11:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 )
        goto LABEL_15;
      goto LABEL_14;
    }
    if ( *v8 == &unk_4F8780C )
      goto LABEL_8;
    ++v8;
  }
  if ( *v8 == &unk_4F8780C )
    goto LABEL_8;
  ++v8;
LABEL_14:
  if ( *v8 == &unk_4F8780C )
    goto LABEL_8;
LABEL_15:
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 156) )
  {
    sub_C8D5F0(a2 + 144, (const void *)(a2 + 160), v7 + 1, 8u, a5, a6);
    v9 = (void **)(*(_QWORD *)(a2 + 144) + 8LL * *(unsigned int *)(a2 + 152));
  }
  *v9 = &unk_4F8780C;
  ++*(_DWORD *)(a2 + 152);
LABEL_9:
  *(_BYTE *)(a2 + 160) = 1;
  nullsub_79();
}

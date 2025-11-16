// Function: sub_29888B0
// Address: 0x29888b0
//
void __fastcall sub_29888B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rcx
  void **v6; // rax
  void **v7; // rsi
  __int64 v8; // rdi
  void **v9; // rdx

  if ( *(_BYTE *)(a1 + 169) )
    sub_BB9660(a2, (__int64)&unk_4F8FC84);
  sub_BB9660(a2, (__int64)&unk_4F8144C);
  v5 = *(unsigned int *)(a2 + 120);
  v6 = *(void ***)(a2 + 112);
  v7 = &v6[v5];
  v8 = (8 * v5) >> 3;
  if ( (8 * v5) >> 5 )
  {
    v9 = &v6[4 * ((8 * v5) >> 5)];
    while ( *v6 != &unk_4F8144C )
    {
      if ( v6[1] == &unk_4F8144C )
      {
        ++v6;
        break;
      }
      if ( v6[2] == &unk_4F8144C )
      {
        v6 += 2;
        break;
      }
      if ( v6[3] == &unk_4F8144C )
      {
        v6 += 3;
        break;
      }
      v6 += 4;
      if ( v9 == v6 )
      {
        v8 = v7 - v6;
        goto LABEL_13;
      }
    }
LABEL_10:
    if ( v7 != v6 )
      goto LABEL_11;
    goto LABEL_16;
  }
LABEL_13:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      if ( v8 != 1 )
        goto LABEL_16;
      goto LABEL_23;
    }
    if ( *v6 == &unk_4F8144C )
      goto LABEL_10;
    ++v6;
  }
  if ( *v6 == &unk_4F8144C )
    goto LABEL_10;
  ++v6;
LABEL_23:
  if ( *v6 == &unk_4F8144C )
    goto LABEL_10;
LABEL_16:
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v3, v4);
    v7 = (void **)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  }
  *v7 = &unk_4F8144C;
  ++*(_DWORD *)(a2 + 120);
LABEL_11:
  nullsub_79();
}

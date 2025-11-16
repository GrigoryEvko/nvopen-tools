// Function: sub_27CDAD0
// Address: 0x27cdad0
//
__int64 __fastcall sub_27CDAD0(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rcx
  void **v6; // rax
  void **v7; // rsi
  __int64 v8; // rdi
  void **v9; // rdx

  sub_BB9630(a2, a2);
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
      if ( v6 == v9 )
      {
        v8 = v7 - v6;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v7 != v6 )
      goto LABEL_9;
    goto LABEL_15;
  }
LABEL_11:
  if ( v8 != 2 )
  {
    if ( v8 != 3 )
    {
      if ( v8 != 1 )
        goto LABEL_15;
      goto LABEL_14;
    }
    if ( *v6 == &unk_4F8144C )
      goto LABEL_8;
    ++v6;
  }
  if ( *v6 == &unk_4F8144C )
    goto LABEL_8;
  ++v6;
LABEL_14:
  if ( *v6 == &unk_4F8144C )
    goto LABEL_8;
LABEL_15:
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v5 + 1, 8u, v3, v4);
    v7 = (void **)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  }
  *v7 = &unk_4F8144C;
  ++*(_DWORD *)(a2 + 120);
LABEL_9:
  sub_BB9660(a2, (__int64)&unk_4F8662C);
  return sub_BB9660(a2, (__int64)&unk_4F89C28);
}

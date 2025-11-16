// Function: sub_2F25C30
// Address: 0x2f25c30
//
void __fastcall sub_2F25C30(__int64 a1, __int64 a2)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rcx
  void **v7; // rax
  void **v8; // rsi
  __int64 v9; // rdi
  void **v10; // rdx

  sub_BB9630(a2, a2);
  sub_BB9660(a2, (__int64)&unk_501FE44);
  v6 = *(unsigned int *)(a2 + 120);
  v7 = *(void ***)(a2 + 112);
  v8 = &v7[v6];
  v9 = (8 * v6) >> 3;
  if ( (8 * v6) >> 5 )
  {
    v10 = &v7[4 * ((8 * v6) >> 5)];
    while ( *v7 != &unk_501FE44 )
    {
      if ( v7[1] == &unk_501FE44 )
      {
        ++v7;
        break;
      }
      if ( v7[2] == &unk_501FE44 )
      {
        v7 += 2;
        break;
      }
      if ( v7[3] == &unk_501FE44 )
      {
        v7 += 3;
        break;
      }
      v7 += 4;
      if ( v7 == v10 )
      {
        v9 = v8 - v7;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v8 != v7 )
      goto LABEL_9;
    goto LABEL_14;
  }
LABEL_11:
  if ( v9 != 2 )
  {
    if ( v9 != 3 )
    {
      if ( v9 != 1 )
        goto LABEL_14;
      goto LABEL_21;
    }
    if ( *v7 == &unk_501FE44 )
      goto LABEL_8;
    ++v7;
  }
  if ( *v7 == &unk_501FE44 )
    goto LABEL_8;
  ++v7;
LABEL_21:
  if ( *v7 == &unk_501FE44 )
    goto LABEL_8;
LABEL_14:
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 124) )
  {
    sub_C8D5F0(a2 + 112, (const void *)(a2 + 128), v6 + 1, 8u, v4, v5);
    v8 = (void **)(*(_QWORD *)(a2 + 112) + 8LL * *(unsigned int *)(a2 + 120));
  }
  *v8 = &unk_501FE44;
  ++*(_DWORD *)(a2 + 120);
LABEL_9:
  sub_2E84680(a1, a2);
}

// Function: sub_22ACB30
// Address: 0x22acb30
//
void __fastcall sub_22ACB30(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  unsigned __int64 *v4; // r15
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rdx
  bool v9; // zf

  ++*(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)(a1 + 68) )
    goto LABEL_6;
  v2 = 4 * (*(_DWORD *)(a1 + 60) - *(_DWORD *)(a1 + 64));
  v3 = *(unsigned int *)(a1 + 56);
  if ( v2 < 0x20 )
    v2 = 32;
  if ( (unsigned int)v3 <= v2 )
  {
    memset(*(void **)(a1 + 48), -1, 8 * v3);
LABEL_6:
    *(_QWORD *)(a1 + 60) = 0;
    goto LABEL_7;
  }
  sub_C8C990(a1 + 40, a2);
LABEL_7:
  v4 = *(unsigned __int64 **)(a1 + 208);
  while ( (unsigned __int64 *)(a1 + 200) != v4 )
  {
    v7 = v4;
    v4 = (unsigned __int64 *)v4[1];
    v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
    *v4 = v8 | *v4 & 7;
    *(_QWORD *)(v8 + 8) = v4;
    *v7 &= 7u;
    v9 = *((_BYTE *)v7 + 76) == 0;
    v7[1] = 0;
    *(v7 - 4) = (unsigned __int64)&unk_4A09CC0;
    if ( v9 )
      _libc_free(v7[7]);
    v5 = v7[5];
    if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
      sub_BD60C0(v7 + 3);
    *(v7 - 4) = (unsigned __int64)&unk_49DB368;
    v6 = *(v7 - 1);
    if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
      sub_BD60C0(v7 - 3);
    j_j___libc_free_0((unsigned __int64)(v7 - 4));
  }
}

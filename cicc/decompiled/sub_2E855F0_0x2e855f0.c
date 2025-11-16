// Function: sub_2E855F0
// Address: 0x2e855f0
//
__int64 __fastcall sub_2E855F0(__int64 a1, __int64 a2)
{
  __int64 **v2; // rbx
  __int64 v3; // r15
  unsigned __int64 v5; // r13
  __int64 *v6; // rcx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  char v10; // cl
  unsigned __int64 v11; // rdx
  char v12; // si
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdi
  char v18; // dl
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  char v23; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 ***)a1;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 == v3 )
    return 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v6 = *v2;
      v7 = **v2;
      if ( !v7 || (v7 & 4) == 0 )
        BUG();
      if ( *(_BYTE *)(*(_QWORD *)(a2 + 8)
                    + 40LL * (unsigned int)(*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 16) + *(_DWORD *)(a2 + 32))
                    + 18) )
        break;
      if ( (__int64 **)v3 == ++v2 )
        goto LABEL_15;
    }
    v8 = v6[3];
    if ( (v8 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
      return -1;
    v10 = *((_BYTE *)v6 + 24);
    v11 = v8 >> 3;
    v12 = v10 & 2;
    if ( (v10 & 6) == 2 || (v10 & 1) != 0 )
    {
      v20 = HIDWORD(v8);
      v21 = HIWORD(v8);
      if ( !v12 )
        v21 = v20;
      v18 = 0;
      v19 = (v21 + 7) >> 3;
    }
    else
    {
      v13 = v8;
      v14 = v8;
      v15 = HIDWORD(v8);
      v16 = v13 >> 8;
      v17 = HIWORD(v14);
      if ( v12 )
        LODWORD(v15) = v17;
      v18 = v11 & 1;
      v19 = ((unsigned __int64)((unsigned __int16)v16 * (unsigned int)v15) + 7) >> 3;
    }
    v22 = v19;
    ++v2;
    v23 = v18;
    v5 += sub_CA1930(&v22);
  }
  while ( (__int64 **)v3 != v2 );
LABEL_15:
  if ( v5 > 0x3FFFFFFFFFFFFFFBLL )
    return 0xBFFFFFFFFFFFFFFELL;
  return v5;
}

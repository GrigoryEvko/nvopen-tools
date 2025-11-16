// Function: sub_2F644B0
// Address: 0x2f644b0
//
__int64 __fastcall sub_2F644B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 result; // rax
  unsigned __int64 v13; // rbx
  __int64 *v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // edi
  unsigned int v17; // esi
  __int64 v18; // rdx

  v6 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 272LL) + 16LL * a5);
  v7 = *v6;
  v8 = v6[1];
  if ( (*(_BYTE *)(a4 + 3) & 0x10) != 0 )
  {
    v7 = ~v7;
    v8 = ~v8;
  }
  v9 = *(_QWORD *)(a2 + 104);
  if ( !v9 )
  {
LABEL_10:
    v13 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    *(_BYTE *)(a4 + 4) |= 1u;
    v14 = (__int64 *)sub_2E09D00((__int64 *)a2, a3 & 0xFFFFFFFFFFFFFFF8LL);
    v15 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
    if ( v14 != (__int64 *)v15 )
    {
      v16 = *(_DWORD *)(v13 + 24);
      v17 = *(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( (unsigned __int64)(v17 | (*v14 >> 1) & 3) <= v16 && v13 == (v14[1] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( (__int64 *)v15 == v14 + 3 )
          goto LABEL_16;
        v17 = *(_DWORD *)((v14[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v14 += 3;
      }
      if ( v17 <= v16 )
      {
        v18 = v14[2];
        result = v14[1] ^ 6;
        if ( (result & 6) != 0 )
        {
          if ( v18 )
            return result;
        }
      }
    }
LABEL_16:
    *(_BYTE *)(a1 + 496) = 1;
    return a1;
  }
  v10 = (a3 >> 1) & 3;
  while ( 1 )
  {
    if ( v8 & *(_QWORD *)(v9 + 120) | v7 & *(_QWORD *)(v9 + 112) )
    {
      v11 = (__int64 *)sub_2E09D00((__int64 *)v9, a3);
      if ( v11 != (__int64 *)(*(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8)) )
      {
        result = *(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3;
        if ( (unsigned int)result <= ((unsigned int)v10 | *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
          return result;
      }
    }
    v9 = *(_QWORD *)(v9 + 104);
    if ( !v9 )
      goto LABEL_10;
  }
}

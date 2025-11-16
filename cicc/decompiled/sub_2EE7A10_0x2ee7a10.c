// Function: sub_2EE7A10
// Address: 0x2ee7a10
//
void __fastcall sub_2EE7A10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // ecx
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rbx
  __int64 v14; // r9
  __int64 v15; // rdx
  _QWORD *v16; // rdx

  v5 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v5 != 1 )
  {
    v6 = 1;
    v7 = *(_QWORD *)(a1 + 32);
    while ( *(_QWORD *)(v7 + 40LL * (v6 + 1) + 24) != a3 )
    {
      v6 += 2;
      if ( v6 == v5 )
        return;
    }
    v8 = v6;
    v9 = *(unsigned int *)(v7 + 40LL * v6 + 8);
    if ( (int)v9 < 0 )
      v10 = *(_QWORD *)(*(_QWORD *)(a4 + 56) + 16 * (v9 & 0x7FFFFFFF) + 8);
    else
      v10 = *(_QWORD *)(*(_QWORD *)(a4 + 304) + 8 * v9);
    if ( !v10
      || (*(_BYTE *)(v10 + 3) & 0x10) == 0 && ((v10 = *(_QWORD *)(v10 + 32)) == 0 || (*(_BYTE *)(v10 + 3) & 0x10) == 0)
      || (v11 = *(_QWORD *)(v10 + 32)) != 0 && (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
    {
      BUG();
    }
    v12 = *(_QWORD *)(v10 + 16);
    v13 = (unsigned int)sub_2EAB0A0(v10) | (unsigned __int64)(v8 << 32);
    v15 = *(unsigned int *)(a2 + 8);
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v15 + 1, 0x10u, v15 + 1, v14);
      v15 = *(unsigned int *)(a2 + 8);
    }
    v16 = (_QWORD *)(*(_QWORD *)a2 + 16 * v15);
    *v16 = v12;
    v16[1] = v13;
    ++*(_DWORD *)(a2 + 8);
  }
}

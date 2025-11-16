// Function: sub_131C810
// Address: 0x131c810
//
__int64 __fastcall sub_131C810(__int64 a1, __int64 a2)
{
  char *v3; // rsi
  __int64 v4; // rdi
  int v5; // ecx
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 result; // rax

  v3 = (char *)&unk_5260DE0;
  v4 = a1 + 80;
  do
  {
    v5 = *(_DWORD *)(v4 + 4);
    v6 = *(int *)(v4 + 8);
    v3 += 40;
    v4 += 28;
    a2 += 4;
    v7 = (v6 << v5) + (1LL << *(_DWORD *)(v4 - 28));
    *((_QWORD *)v3 - 5) = v7;
    v8 = (int)(*(_DWORD *)(v4 - 12) << 12);
    *((_QWORD *)v3 - 4) = v8;
    LODWORD(v8) = v8 / v7;
    *((_DWORD *)v3 - 6) = v8;
    *((_DWORD *)v3 - 5) = *(_DWORD *)(a2 - 4);
    v9 = (unsigned int)v8;
    result = (unsigned int)(v8 + 63) >> 6;
    *((_QWORD *)v3 - 2) = v9;
    *((_QWORD *)v3 - 1) = result;
  }
  while ( v3 != (char *)&unk_5260DE0 + 1440 );
  return result;
}

// Function: sub_1A0F420
// Address: 0x1a0f420
//
void __fastcall sub_1A0F420(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  unsigned __int64 v3; // rax
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rdx
  __int64 v7; // r15
  unsigned __int64 v8; // [rsp+8h] [rbp-38h]

  for ( i = *(_QWORD *)(a1 + 80); a1 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    v7 = i - 24;
    if ( !i )
      v7 = 0;
    if ( sub_157EBE0(v7) )
      break;
    v3 = sub_157EBA0(v7);
    if ( *(_BYTE *)(v3 + 16) == 25
      && *(_BYTE *)(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)) + 16LL) != 9 )
    {
      v6 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 12) )
      {
        v8 = v3;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v4, v5);
        v6 = *(unsigned int *)(a2 + 8);
        v3 = v8;
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v6) = v3;
      ++*(_DWORD *)(a2 + 8);
    }
  }
}

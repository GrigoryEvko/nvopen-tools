// Function: sub_20E8320
// Address: 0x20e8320
//
__int64 __fastcall sub_20E8320(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 i; // rcx
  __int64 v7; // rdx

  v2 = 0;
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 96LL) + 10LL * a2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
  v5 = v4 + 16LL * *(unsigned __int16 *)(v3 + 2);
  for ( i = v4 + 16LL * *(unsigned __int16 *)(v3 + 4); v5 != i; v2 = v7 | (v2 << 16) )
  {
    v7 = *(unsigned int *)(v5 + 4);
    v5 += 16;
  }
  return v2;
}

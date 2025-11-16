// Function: sub_2E86600
// Address: 0x2e86600
//
void __fastcall sub_2E86600(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rax

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v2 = a1;
    if ( (*(_BYTE *)a1 & 4) == 0 && (*(_BYTE *)(a1 + 44) & 8) != 0 )
    {
      do
        v2 = *(_QWORD *)(v2 + 8);
      while ( (*(_BYTE *)(v2 + 44) & 8) != 0 );
    }
    v3 = *(unsigned __int64 **)(v2 + 8);
    if ( (unsigned __int64 *)a1 != v3 && a2 != v3 )
    {
      sub_2E310C0((__int64 *)(a2[3] + 40), (__int64 *)(*(_QWORD *)(a1 + 24) + 40LL), a1, *(_QWORD *)(v2 + 8));
      v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v3;
      *v3 = *v3 & 7 | *(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = *a2;
      *(_QWORD *)(v4 + 8) = a2;
      v5 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)a1 = v5 | *(_QWORD *)a1 & 7LL;
      *(_QWORD *)(v5 + 8) = a1;
      *a2 = v4 | *a2 & 7;
    }
  }
}

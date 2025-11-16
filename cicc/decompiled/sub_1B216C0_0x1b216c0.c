// Function: sub_1B216C0
// Address: 0x1b216c0
//
void __fastcall sub_1B216C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 *v3; // r13
  __int64 v4; // rsi

  if ( byte_4FB6BE0 )
  {
    sub_1B205A0(a1);
    v1 = *(_QWORD *)(*(_QWORD *)(a1 + 488) + 16LL);
    v2 = *(__int64 **)(v1 + 48);
    v3 = &v2[*(unsigned int *)(v1 + 56)];
    while ( v3 != v2 )
    {
      v4 = *v2++;
      sub_1B1FC20(a1, v4, v4);
    }
  }
}

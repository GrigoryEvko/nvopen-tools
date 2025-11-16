// Function: sub_86F5D0
// Address: 0x86f5d0
//
unsigned int *__fastcall sub_86F5D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  char v3; // al
  __int64 v5; // rax

  v1 = sub_86B2C0(1);
  *(_QWORD *)(v1 + 40) = a1;
  v2 = v1;
  v3 = *(_BYTE *)(a1 + 40);
  if ( v3 == 17 )
  {
    *(_QWORD *)(v2 + 48) = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  }
  else if ( v3 == 22 && !*(_BYTE *)(a1 + 72) && (unsigned int)sub_8D4070(*(_QWORD *)(*(_QWORD *)(a1 + 80) + 120LL)) )
  {
    v5 = *(_QWORD *)(a1 + 80);
    *(_BYTE *)(v2 + 56) |= 1u;
    *(_QWORD *)(v2 + 48) = v5;
  }
  *(_BYTE *)(v2 + 56) = (2 * sub_86D9F0()) | *(_BYTE *)(v2 + 56) & 0xFD;
  return sub_86CBE0(v2);
}

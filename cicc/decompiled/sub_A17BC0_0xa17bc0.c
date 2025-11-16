// Function: sub_A17BC0
// Address: 0xa17bc0
//
void __fastcall sub_A17BC0(__int64 a1)
{
  int v1; // ecx
  int v2; // r12d
  unsigned int v3; // ecx
  int v4; // r12d
  _QWORD *v5; // r13
  __int64 v6; // rax
  unsigned int v7; // edx
  int v8; // eax

  sub_A17B10(a1, 0x42u, 8);
  sub_A17B10(a1, 0x43u, 8);
  sub_A17B10(a1, 0, 4);
  sub_A17B10(a1, 0xCu, 4);
  sub_A17B10(a1, 0xEu, 4);
  v1 = *(_DWORD *)(a1 + 48);
  v2 = 13 << v1;
  v3 = v1 + 4;
  v4 = *(_DWORD *)(a1 + 52) | v2;
  *(_DWORD *)(a1 + 52) = v4;
  if ( v3 > 0x1F )
  {
    v5 = *(_QWORD **)(a1 + 24);
    v6 = v5[1];
    if ( (unsigned __int64)(v6 + 4) > v5[2] )
    {
      sub_C8D290(*(_QWORD *)(a1 + 24), v5 + 3, v6 + 4, 1);
      v6 = v5[1];
    }
    *(_DWORD *)(*v5 + v6) = v4;
    v7 = 0;
    v5[1] += 4LL;
    v8 = *(_DWORD *)(a1 + 48);
    if ( v8 )
      v7 = 0xDu >> (32 - v8);
    *(_DWORD *)(a1 + 52) = v7;
    *(_DWORD *)(a1 + 48) = ((_BYTE)v8 + 4) & 0x1F;
  }
  else
  {
    *(_DWORD *)(a1 + 48) = v3;
  }
}

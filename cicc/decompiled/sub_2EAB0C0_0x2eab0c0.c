// Function: sub_2EAB0C0
// Address: 0x2eab0c0
//
void __fastcall sub_2EAB0C0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13

  if ( *(_DWORD *)(a1 + 8) != a2 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    *(_BYTE *)(a1 + 3) &= ~0x80u;
    if ( v2 && (v3 = *(_QWORD *)(v2 + 24)) != 0 && (v4 = *(_QWORD *)(v3 + 32)) != 0 )
    {
      v5 = *(_QWORD *)(v4 + 32);
      sub_2EBEB60(v5, a1);
      *(_DWORD *)(a1 + 8) = a2;
      sub_2EBEAE0(v5, a1);
    }
    else
    {
      *(_DWORD *)(a1 + 8) = a2;
    }
  }
}

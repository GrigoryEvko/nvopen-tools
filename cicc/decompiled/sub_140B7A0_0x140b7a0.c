// Function: sub_140B7A0
// Address: 0x140b7a0
//
__int64 __fastcall sub_140B7A0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int v5; // edi
  _QWORD *v6; // rax
  unsigned __int64 v8; // rax
  unsigned int v9; // ecx
  unsigned __int64 v10; // rdx

  v5 = *(_DWORD *)(a3 + 8);
  v6 = *(_QWORD **)a3;
  if ( *(_BYTE *)(a2 + 17) && a4 )
  {
    if ( v5 > 0x40 )
      v6 = (_QWORD *)*v6;
    v8 = a4 * (((unsigned __int64)v6 + a4 - 1) / a4);
    v9 = *(_DWORD *)(a2 + 20);
    *(_DWORD *)(a1 + 8) = v9;
    if ( v9 > 0x40 )
    {
      sub_16A4EF0(a1, v8, 0);
      return a1;
    }
    else
    {
      v10 = v8 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9);
      *(_QWORD *)a1 = v10;
      return a1;
    }
  }
  else
  {
    *(_QWORD *)a1 = v6;
    *(_DWORD *)(a1 + 8) = v5;
    *(_DWORD *)(a3 + 8) = 0;
    return a1;
  }
}

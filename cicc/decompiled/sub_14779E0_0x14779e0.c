// Function: sub_14779E0
// Address: 0x14779e0
//
__int64 __fastcall sub_14779E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v7; // eax

  v3 = sub_1477920(a2, a3, 0);
  v4 = *((_DWORD *)v3 + 2);
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 > 0x40 )
  {
    sub_16A4FD0(a1, v3);
    v7 = *((_DWORD *)v3 + 6);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 = *v3;
    v5 = *((_DWORD *)v3 + 6);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)(a1 + 16) = v3[2];
      return a1;
    }
  }
  sub_16A4FD0(a1 + 16, v3 + 2);
  return a1;
}

// Function: sub_14CE830
// Address: 0x14ce830
//
void __fastcall sub_14CE830(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD v5[2]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v6; // [rsp-38h] [rbp-38h]
  int v7; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)(a1 + 184) )
  {
    v5[0] = 6;
    v5[1] = 0;
    v6 = a2;
    if ( a2 != 0 && a2 != -8 && a2 != -16 )
      sub_164C220(v5);
    v7 = -1;
    v2 = *(_DWORD *)(a1 + 16);
    if ( v2 >= *(_DWORD *)(a1 + 20) )
    {
      sub_14CB640(a1 + 8, 0);
      v2 = *(_DWORD *)(a1 + 16);
    }
    v3 = *(_QWORD *)(a1 + 8) + 32LL * v2;
    if ( v3 )
    {
      *(_QWORD *)v3 = 6;
      *(_QWORD *)(v3 + 8) = 0;
      v4 = v6;
      *(_QWORD *)(v3 + 16) = v6;
      if ( v4 != -8 && v4 != 0 && v4 != -16 )
        sub_1649AC0(v3, v5[0] & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v3 + 24) = v7;
      v2 = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v2 + 1;
    if ( v6 != -8 && v6 != 0 && v6 != -16 )
      sub_1649B30(v5);
    sub_14CDA00(a1, a2);
  }
}

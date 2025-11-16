// Function: sub_39388E0
// Address: 0x39388e0
//
__int64 __fastcall sub_39388E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-30h] BYREF
  char v6; // [rsp+20h] [rbp-20h]

  sub_16C2E90((__int64)v5, a2, 0xFFFFFFFFFFFFFFFFLL, 1);
  if ( (v6 & 1) == 0 || !LODWORD(v5[0]) )
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = v5[0];
    return a1;
  }
  sub_16BCB40(&v4, v5[0], v5[1]);
  v3 = v4;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v3 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v6 & 1) != 0 || !v5[0] )
    return a1;
  (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
  return a1;
}

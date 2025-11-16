// Function: sub_E7F9C0
// Address: 0xe7f9c0
//
__int64 __fastcall sub_E7F9C0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rax
  unsigned __int8 v4; // cl
  unsigned __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  const char *v7; // [rsp+10h] [rbp-40h] BYREF
  char v8; // [rsp+30h] [rbp-20h]
  char v9; // [rsp+31h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 3536) )
  {
    v9 = 1;
    v6 = 0;
    v7 = ".gnu.attributes";
    v8 = 3;
    sub_E7EC80((_QWORD *)a1, (__int64)"gnu", 3, (size_t *)&v7, 1879048181, &v6, a1 + 3528);
  }
  v1 = *(_QWORD *)(a1 + 288);
  if ( v1 )
  {
    v2 = *(unsigned int *)(*(_QWORD *)(a1 + 296) + 368LL);
    if ( (_DWORD)v2 )
    {
      v3 = *(_QWORD *)(v1 + 8);
      if ( (*(_BYTE *)(v3 + 48) & 2) != 0 )
      {
        _BitScanReverse64(&v2, v2);
        v4 = 63 - (v2 ^ 0x3F);
        if ( v4 > *(_BYTE *)(v3 + 32) )
          *(_BYTE *)(v3 + 32) = v4;
      }
    }
  }
  sub_E7E9B0(a1);
  sub_E8ACF0(a1, 0);
  return sub_E8AC70(a1);
}

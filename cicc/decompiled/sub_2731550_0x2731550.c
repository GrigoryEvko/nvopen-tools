// Function: sub_2731550
// Address: 0x2731550
//
unsigned __int64 __fastcall sub_2731550(
        unsigned __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 result; // rax

  v6 = a1[1];
  if ( v6 == a1[2] )
    return sub_2731260(a1, a1[1], a2, a4);
  if ( v6 )
  {
    *(_QWORD *)v6 = v6 + 16;
    *(_QWORD *)(v6 + 8) = 0x800000000LL;
    if ( *(_DWORD *)(a2 + 8) )
      sub_272D8A0(v6, (char **)a2, a3, a4, a5, a6);
    *(_QWORD *)(v6 + 144) = *(_QWORD *)(a2 + 144);
    *(_QWORD *)(v6 + 152) = *(_QWORD *)(a2 + 152);
    result = *(unsigned int *)(a2 + 160);
    *(_DWORD *)(v6 + 160) = result;
    v6 = a1[1];
  }
  a1[1] = v6 + 168;
  return result;
}

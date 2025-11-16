// Function: sub_B59530
// Address: 0xb59530
//
__int64 __fastcall sub_B59530(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // [rsp+20h] [rbp-30h] BYREF
  char v4; // [rsp+30h] [rbp-20h]

  v1 = *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  sub_AF47B0(
    (__int64)&v3,
    *(unsigned __int64 **)(*(_QWORD *)(v1 + 24) + 16LL),
    *(unsigned __int64 **)(*(_QWORD *)(v1 + 24) + 24LL));
  if ( v4 )
    return v3;
  else
    return sub_AF3FE0(*(_QWORD *)(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 24LL));
}

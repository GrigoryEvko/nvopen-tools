// Function: sub_34B23E0
// Address: 0x34b23e0
//
__int64 __fastcall sub_34B23E0(
        __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6)
{
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // rax

  if ( sub_B92180(*a2)
    && ((v9 = sub_B92180(*a2), v10 = *(_BYTE *)(v9 - 16), (v10 & 2) != 0)
      ? (v11 = *(_QWORD *)(v9 - 32))
      : (v11 = v9 - 16 - 8LL * ((v10 >> 2) & 0xF)),
        *(_DWORD *)(*(_QWORD *)(v11 + 40) + 32LL)) )
  {
    return sub_34AF4A0(a1, a2, a4, a5, a6);
  }
  else
  {
    return 0;
  }
}

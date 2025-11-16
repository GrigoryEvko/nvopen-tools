// Function: sub_693DC0
// Address: 0x693dc0
//
__int64 __fastcall sub_693DC0(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r13
  __int64 v4; // rdx
  unsigned int v5; // r13d
  _DWORD *v6; // r12
  __m128i v7[3]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 64);
  if ( (unsigned int)sub_89A370(v2) )
  {
    *a2 = 1;
    return 0;
  }
  else
  {
    v4 = **(_QWORD **)(*(_QWORD *)(**(_QWORD **)(a1 + 56) + 88LL) + 32LL);
    v7[0] = 0u;
    v5 = sub_6F1C10(a1, v2, v4, (unsigned int)v7, 0, 0, (__int64)a2, 0);
    if ( *a2 )
    {
      v6 = sub_67D9D0(0xC3Eu, (_DWORD *)(a1 + 28));
      sub_67E370((__int64)v6, v7);
      sub_685910((__int64)v6, (FILE *)v7);
    }
    else
    {
      sub_67E3D0(v7);
    }
    return v5;
  }
}

// Function: sub_2AB9310
// Address: 0x2ab9310
//
__int64 __fastcall sub_2AB9310(_QWORD *a1, unsigned int a2)
{
  unsigned int v3; // r13d
  __int64 v4; // r9
  __int64 *v5; // r8
  unsigned __int8 *v6; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+8h] [rbp-28h]

  if ( (unsigned __int8)sub_2AB8E90((__int64)a1) )
  {
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[55] + 56LL) + 16LL) + 216LL) == 0xFFFFFFFFLL )
    {
      LODWORD(v7) = -1;
      BYTE4(v7) = 1;
      return v7;
    }
    else
    {
      v7 = sub_2AA7E40(a1[61], a1[56]);
      v3 = a2 / (unsigned int)v7;
      if ( a2 < (unsigned int)v7 )
      {
        v4 = a1[52];
        v5 = (__int64 *)a1[60];
        v6 = 0;
        sub_2AB8CE0(
          "Max legal vector width too small, scalable vectorization unfeasible.",
          0x44u,
          (__int64)"ScalableVFUnfeasible",
          20,
          v5,
          v4,
          &v6);
        sub_9C6650(&v6);
      }
      LODWORD(v7) = v3;
      BYTE4(v7) = 1;
      return v7;
    }
  }
  else
  {
    LODWORD(v7) = 0;
    BYTE4(v7) = 1;
    return v7;
  }
}

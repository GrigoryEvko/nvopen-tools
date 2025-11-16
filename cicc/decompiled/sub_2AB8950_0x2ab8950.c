// Function: sub_2AB8950
// Address: 0x2ab8950
//
__int64 __fastcall sub_2AB8950(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // rax
  unsigned int v3; // r12d

  v1 = **(unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 8LL);
  if ( (_BYTE)v1 )
  {
    sub_2AB8760(
      (__int64)"Runtime ptr check is required with -Os/-Oz",
      42,
      "runtime pointer checks needed. Enable vectorization of this loop with '#pragma clang loop vectorize(enable)' when "
      "compiling with -Os/-Oz",
      0x88u,
      (__int64)"CantVersionLoopWithOptForSize",
      29,
      *(__int64 **)(a1 + 480),
      *(_QWORD *)(a1 + 416),
      0);
    return v1;
  }
  else
  {
    v2 = sub_D9B120(*(_QWORD *)(a1 + 424));
    v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
    if ( (_BYTE)v3 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 136LL) )
      {
        sub_2AB8760(
          (__int64)"Runtime stride check for small trip count",
          41,
          "runtime stride == 1 checks needed. Enable vectorization of this loop without such check by compiling with -Os/-Oz",
          0x71u,
          (__int64)"CantVersionLoopWithOptForSize",
          29,
          *(__int64 **)(a1 + 480),
          *(_QWORD *)(a1 + 416),
          0);
        return v3;
      }
      else
      {
        return v1;
      }
    }
    else
    {
      sub_2AB8760(
        (__int64)"Runtime SCEV check is required with -Os/-Oz",
        43,
        "runtime SCEV checks needed. Enable vectorization of this loop with '#pragma clang loop vectorize(enable)' when c"
        "ompiling with -Os/-Oz",
        0x85u,
        (__int64)"CantVersionLoopWithOptForSize",
        29,
        *(__int64 **)(a1 + 480),
        *(_QWORD *)(a1 + 416),
        0);
      return 1;
    }
  }
}

// Function: sub_AA8900
// Address: 0xaa8900
//
__int64 __fastcall sub_AA8900(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // edx
  __int64 v4; // r12
  int v5; // edx

  if ( !(unsigned __int8)sub_B2F6B0(a1)
    && *(_BYTE *)(a1 + 32) >> 6 != 2
    && (*(_BYTE *)a1 != 3
     || ((v2 = *(_QWORD *)(a1 + 24), v3 = *(unsigned __int8 *)(v2 + 8), (_BYTE)v3 == 12)
      || (unsigned __int8)v3 <= 3u
      || (_BYTE)v3 == 5
      || (v3 & 0xFB) == 0xA
      || (v3 & 0xFD) == 4
      || ((unsigned __int8)(*(_BYTE *)(v2 + 8) - 15) <= 3u || v3 == 20)
      && (unsigned __int8)sub_BCEBA0(*(_QWORD *)(a1 + 24), 0))
     && !(unsigned __int8)sub_BCADB0(v2))
    && !(unsigned __int8)sub_B2F6B0(a2)
    && *(_BYTE *)(a2 + 32) >> 6 != 2
    && (*(_BYTE *)a2 != 3
     || ((v4 = *(_QWORD *)(a2 + 24), v5 = *(unsigned __int8 *)(v4 + 8), (_BYTE)v5 == 12)
      || (unsigned __int8)v5 <= 3u
      || (_BYTE)v5 == 5
      || (v5 & 0xFB) == 0xA
      || (v5 & 0xFD) == 4
      || ((unsigned __int8)(*(_BYTE *)(v4 + 8) - 15) <= 3u || v5 == 20) && (unsigned __int8)sub_BCEBA0(v4, 0))
     && !(unsigned __int8)sub_BCADB0(v4)) )
  {
    return 33;
  }
  else
  {
    return 42;
  }
}

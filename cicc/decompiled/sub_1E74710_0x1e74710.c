// Function: sub_1E74710
// Address: 0x1e74710
//
void __fastcall sub_1E74710(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  int v6; // r15d
  int v7; // eax

  v5 = *(_QWORD *)(a2 + 16);
  if ( !v5
    || (v6 = sub_1E72BB0(a1 + 136, v5),
        v7 = sub_1E72BB0(a1 + 136, *(_QWORD *)(a3 + 16)),
        !(unsigned __int8)sub_1E738C0(v7, v6, a3, a2, 5u))
    && !(unsigned __int8)sub_1E738F0(
                           *(_QWORD *)(a3 + 16) == *(_QWORD *)(*(_QWORD *)(a1 + 128) + 2264LL),
                           *(_QWORD *)(a2 + 16) == *(_QWORD *)(*(_QWORD *)(a1 + 128) + 2264LL),
                           a3,
                           a2,
                           6u)
    && !(unsigned __int8)sub_1E738C0(*(_DWORD *)(a3 + 40), *(_DWORD *)(a2 + 40), a3, a2, 9u)
    && !(unsigned __int8)sub_1E738F0(*(_DWORD *)(a3 + 44), *(_DWORD *)(a2 + 44), a3, a2, 0xAu)
    && (!*(_BYTE *)a2 || !(unsigned __int8)sub_1E73920(a3, a2, (_DWORD *)(a1 + 136)))
    && *(_DWORD *)(*(_QWORD *)(a3 + 16) + 192LL) < *(_DWORD *)(*(_QWORD *)(a2 + 16) + 192LL) )
  {
    *(_BYTE *)(a3 + 24) = 16;
  }
}

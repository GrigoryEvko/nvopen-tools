// Function: sub_1F5BDD0
// Address: 0x1f5bdd0
//
__int64 __fastcall sub_1F5BDD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 280LL)
     + 24LL
     * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
      + *(_DWORD *)(*(_QWORD *)(a1 + 248) + 288LL)
      * (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 264LL) - *(_QWORD *)(*(_QWORD *)(a1 + 248) + 256LL)) >> 3));
  return sub_1E091A0(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 56LL), *(_DWORD *)(v2 + 4) >> 3, *(_DWORD *)(v2 + 8) >> 3);
}

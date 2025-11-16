// Function: sub_89D460
// Address: 0x89d460
//
_BOOL8 __fastcall sub_89D460(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 *v12; // r12
  __int64 *v13; // rbx

  v2 = 776LL * dword_4F04C64;
  v3 = qword_4F04C68[0] + v2 - 776;
  if ( *(_BYTE *)(v3 + 4) != 6 )
    return 0;
  v5 = a1;
  if ( (*(_DWORD *)&word_4D04A10 & 0x10001) != 0 )
    return 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + v2 + 6) & 2) == 0 )
    return 0;
  if ( *(_BYTE *)(a1 + 80) != 19 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 266LL) & 1) != 0 )
    return 0;
  if ( *(_DWORD *)(a1 + 40) == *(_DWORD *)v3 )
    return 0;
  v6 = *(__int64 **)(*(_QWORD *)(sub_6724F0() + 168) + 160LL);
  if ( !v6 )
    return 0;
  v7 = *v6;
  if ( *(_BYTE *)(*v6 + 80) != 19 )
    return 0;
  v8 = *(_QWORD *)(v7 + 88);
  if ( *(_QWORD *)(v8 + 88) && (*(_BYTE *)(v8 + 160) & 1) == 0 )
    v7 = *(_QWORD *)(v8 + 88);
  v9 = sub_892920(v7);
  v10 = *(_QWORD *)(v5 + 88);
  v11 = v9;
  if ( *(_QWORD *)(v10 + 88) && (*(_BYTE *)(v10 + 160) & 1) == 0 )
    v5 = *(_QWORD *)(v10 + 88);
  if ( v9 == sub_892920(v5)
    && (v12 = *(__int64 **)(a2 + 192),
        v13 = *(__int64 **)(*(_QWORD *)(v11 + 88) + 32LL),
        (unsigned int)sub_89B3C0(*v13, *v12, 0, 2u, dword_4F07508, 3u)) )
  {
    return (unsigned int)sub_739400(*(__int128 **)(v13[4] + 16), *(__int128 **)(v12[4] + 16)) == 0;
  }
  else
  {
    return 1;
  }
}

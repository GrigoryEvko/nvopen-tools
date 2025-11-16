// Function: sub_2ECE190
// Address: 0x2ece190
//
__int64 __fastcall sub_2ECE190(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  bool v3; // zf
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 result; // rax

  v2 = a2 + 600;
  v3 = *(_BYTE *)(a1 + 37) == 0;
  *(_QWORD *)(a1 + 136) = a2;
  *(_QWORD *)(a1 + 16) = a2 + 600;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
  if ( !v3 )
  {
    sub_2ECB100(a2);
    v2 = *(_QWORD *)(a1 + 16);
    a2 = *(_QWORD *)(a1 + 136);
  }
  sub_2EC8A00(a1 + 40, a2, v2);
  sub_2ECDD50(a1 + 144, *(_QWORD *)(a1 + 136), *(_QWORD *)(a1 + 16), a1 + 40, v5, v6);
  sub_2ECDD50(a1 + 864, *(_QWORD *)(a1 + 136), *(_QWORD *)(a1 + 16), a1 + 40, v7, v8);
  v9 = *(_QWORD *)(a1 + 16);
  v10 = v9 + 80;
  result = sub_2FF7B90(v9);
  if ( !(_BYTE)result )
    v10 = 0;
  if ( !*(_QWORD *)(a1 + 296) )
  {
    result = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 136) + 16LL) + 1040LL))(
               *(_QWORD *)(*(_QWORD *)(a1 + 136) + 16LL),
               v10);
    *(_QWORD *)(a1 + 296) = result;
  }
  if ( !*(_QWORD *)(a1 + 1016) )
  {
    result = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 136) + 16LL) + 1040LL))(
               *(_QWORD *)(*(_QWORD *)(a1 + 136) + 16LL),
               v10);
    *(_QWORD *)(a1 + 1016) = result;
  }
  *(_QWORD *)(a1 + 1600) = 0;
  *(_QWORD *)(a1 + 1648) = 0;
  return result;
}

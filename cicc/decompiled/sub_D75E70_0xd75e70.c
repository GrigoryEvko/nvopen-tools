// Function: sub_D75E70
// Address: 0xd75e70
//
__int64 __fastcall sub_D75E70(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rbp
  __int64 v4; // [rsp-28h] [rbp-28h] BYREF
  __int64 v5; // [rsp-20h] [rbp-20h]
  __int64 v6; // [rsp-8h] [rbp-8h]

  if ( !**(_BYTE **)a1 )
    return 0;
  v6 = v2;
  sub_B8BA60((__int64)&v4, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL), *(_QWORD *)(a1 + 8), (__int64)&unk_4F87F20, a2);
  return (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F87F20) + 176;
}

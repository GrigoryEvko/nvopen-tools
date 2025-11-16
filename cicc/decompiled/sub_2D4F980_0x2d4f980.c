// Function: sub_2D4F980
// Address: 0x2d4f980
//
__int64 __fastcall sub_2D4F980(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  _QWORD v6[4]; // [rsp+0h] [rbp-20h] BYREF

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  if ( !v2 )
    return 0;
  v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_5027190);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(v3 + 256);
  v6[0] = 0;
  v6[1] = 0;
  return sub_2D4F770((__int64)v6, a2, v4);
}

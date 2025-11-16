// Function: sub_30CC550
// Address: 0x30cc550
//
__int64 __fastcall sub_30CC550(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // [rsp+8h] [rbp-28h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h]
  __int64 v8; // [rsp+18h] [rbp-18h]

  v2 = *(_QWORD *)(a1 - 32);
  if ( v2 )
  {
    if ( *(_BYTE *)v2 )
    {
      v2 = 0;
    }
    else if ( *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    {
      v2 = 0;
    }
  }
  v6 = a2;
  v3 = sub_BC1CD0(a2, &unk_4F89C30, v2);
  v7 = sub_30D6600(a1, v2, v3 + 8, sub_30CA280, &v6);
  result = 0;
  v8 = v5;
  if ( (_BYTE)v5 )
    return 1 - ((unsigned int)(v7 == 0) - 1);
  return result;
}

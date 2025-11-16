// Function: sub_31A91F0
// Address: 0x31a91f0
//
__int64 __fastcall sub_31A91F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // bl
  int v7; // eax
  __int64 result; // rax
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = a4;
  v7 = *(_DWORD *)(a1 + 40);
  v9[0] = a3;
  if ( v7 == -1 )
  {
    if ( (unsigned __int8)sub_F6E590(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6) )
      goto LABEL_7;
    v7 = *(_DWORD *)(a1 + 40);
  }
  if ( !v7 )
  {
LABEL_7:
    sub_31A8E80(a1);
    return 0;
  }
  if ( !v6 )
    goto LABEL_4;
  if ( v7 == -1 )
  {
    if ( !(unsigned __int8)sub_F6E590(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6) )
    {
      v7 = *(_DWORD *)(a1 + 40);
      goto LABEL_9;
    }
    goto LABEL_7;
  }
LABEL_9:
  if ( v7 != 1 )
    goto LABEL_7;
LABEL_4:
  result = 1;
  if ( *(_DWORD *)(a1 + 56) == 1 )
  {
    sub_31A8E90(*(__int64 **)(a1 + 112), a1, v9);
    return 0;
  }
  return result;
}

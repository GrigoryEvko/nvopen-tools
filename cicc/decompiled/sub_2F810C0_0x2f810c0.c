// Function: sub_2F810C0
// Address: 0x2f810c0
//
__int64 __fastcall sub_2F810C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501EACC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501EACC);
  v9 = 0;
  v7 = v5 + 200;
  v8 = *(_QWORD *)(a2 + 32);
  result = *(unsigned __int8 *)(v8 + 48);
  if ( (_BYTE)result )
    return sub_2F80AA0(&v7, *(_QWORD *)(a2 + 16));
  return result;
}

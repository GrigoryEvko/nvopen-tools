// Function: sub_22F33D0
// Address: 0x22f33d0
//
__int64 __fastcall sub_22F33D0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  v1 = sub_22F59B0(a1[1], *(unsigned __int16 *)(*a1 + 58LL));
  if ( !v1 )
    return *a1;
  v3 = v1;
  v4 = sub_22F59B0(v2, *(unsigned __int16 *)(v1 + 58));
  v6 = v4;
  if ( v4 )
  {
    v7 = sub_22F59B0(v5, *(unsigned __int16 *)(v4 + 58));
    v3 = v7;
    if ( v7 )
    {
      v11[0] = sub_22F59B0(v8, *(unsigned __int16 *)(v7 + 58));
      v11[1] = v9;
      if ( v11[0] )
        return sub_22F33D0(v11);
    }
    else
    {
      return v6;
    }
  }
  return v3;
}

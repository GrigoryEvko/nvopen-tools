// Function: sub_168F990
// Address: 0x168f990
//
__int64 __fastcall sub_168F990(_QWORD *a1)
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

  v1 = sub_1691920(a1[1], *(unsigned __int16 *)(*a1 + 42LL));
  if ( !v1 )
    return *a1;
  v3 = v1;
  v4 = sub_1691920(v2, *(unsigned __int16 *)(v1 + 42));
  v6 = v4;
  if ( v4 )
  {
    v7 = sub_1691920(v5, *(unsigned __int16 *)(v4 + 42));
    v3 = v7;
    if ( v7 )
    {
      v11[0] = sub_1691920(v8, *(unsigned __int16 *)(v7 + 42));
      v11[1] = v9;
      if ( v11[0] )
        return sub_168F990(v11);
    }
    else
    {
      return v6;
    }
  }
  return v3;
}

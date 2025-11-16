// Function: sub_BB8160
// Address: 0xbb8160
//
__int64 __fastcall sub_BB8160(__int64 a1, int *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  _BOOL4 v5; // r8d
  __int64 v6; // rbx
  _BOOL4 v7; // [rsp+Ch] [rbp-34h]

  result = sub_BB80C0(a1, a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != a1 + 8 )
      v5 = *a2 < *(_DWORD *)(v3 + 32);
    v7 = v5;
    v6 = sub_22077B0(40);
    *(_DWORD *)(v6 + 32) = *a2;
    sub_220F040(v7, v6, v4, a1 + 8);
    ++*(_QWORD *)(a1 + 40);
    return v6;
  }
  return result;
}

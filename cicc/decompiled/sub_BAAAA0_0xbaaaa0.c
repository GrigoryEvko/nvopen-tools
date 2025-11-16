// Function: sub_BAAAA0
// Address: 0xbaaaa0
//
__int64 __fastcall sub_BAAAA0(__int64 **a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  int v4; // eax
  __int64 v5; // rdx
  double v6; // xmm0_8
  unsigned __int8 *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rcx

  result = sub_BAA6A0((__int64)a1, 0);
  if ( result )
  {
    result = sub_BC9AA0(result);
    v3 = result;
    if ( result )
    {
      if ( *(_DWORD *)result == 2
        && *(_BYTE *)(result + 72)
        && (v4 = *(_DWORD *)(result + 64), v5 = *(_QWORD *)(a2 + 520), v4) )
      {
        if ( v5 < 0 )
        {
          v9 = *(_QWORD *)(a2 + 520) & 1LL | (*(_QWORD *)(a2 + 520) >> 1);
          v6 = (double)(int)v9 + (double)(int)v9;
        }
        else
        {
          v6 = (double)(int)v5;
        }
        *(double *)(v3 + 80) = v6 / (double)v4;
        v7 = (unsigned __int8 *)sub_BCA5C0(v3, *a1, 1, 1);
        sub_BAA660(a1, v7, 2);
        v8 = *(_QWORD *)(v3 + 8);
        if ( !v8 )
          return j_j___libc_free_0(v3, 88);
      }
      else
      {
        v8 = *(_QWORD *)(v3 + 8);
        if ( !v8 )
          return j_j___libc_free_0(v3, 88);
      }
      j_j___libc_free_0(v8, *(_QWORD *)(v3 + 24) - v8);
      return j_j___libc_free_0(v3, 88);
    }
  }
  return result;
}

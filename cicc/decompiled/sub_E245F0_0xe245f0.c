// Function: sub_E245F0
// Address: 0xe245f0
//
__int64 __fastcall sub_E245F0(__int64 **a1)
{
  __int64 v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 *v4; // rax
  __int64 *v5; // r12
  __int64 *v6; // rdx

  v1 = **a1;
  v2 = (v1 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a1)[1] = v2 - v1 + 64;
  if ( (*a1)[1] <= (unsigned __int64)(*a1)[2] )
  {
    result = 0;
    if ( !v2 )
      return result;
    result = v2;
    goto LABEL_4;
  }
  v4 = (__int64 *)sub_22077B0(32);
  v5 = v4;
  if ( v4 )
  {
    *v4 = 0;
    v4[1] = 0;
    v4[2] = 0;
    v4[3] = 0;
  }
  result = sub_2207820(4096);
  v6 = *a1;
  *a1 = v5;
  *v5 = result;
  v5[3] = (__int64)v6;
  v5[2] = 4096;
  v5[1] = 64;
  if ( result )
  {
LABEL_4:
    *(_DWORD *)(result + 8) = 21;
    *(_QWORD *)(result + 16) = 0;
    *(_DWORD *)(result + 24) = 0;
    *(_QWORD *)result = &unk_49E0F38;
    *(_DWORD *)(result + 56) = 0;
    *(_BYTE *)(result + 60) = 0;
  }
  return result;
}

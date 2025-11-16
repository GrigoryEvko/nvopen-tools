// Function: sub_88F310
// Address: 0x88f310
//
__int64 __fastcall sub_88F310(__int64 a1)
{
  __int64 *v1; // r12
  __int64 *v2; // r13
  __int64 v3; // r14
  __int64 result; // rax
  __int64 v5; // rdx

  v1 = *(__int64 **)(a1 + 32);
  v2 = *(__int64 **)(a1 + 40);
  v3 = *v1;
  result = sub_7AE2C0(16, *(_DWORD *)(a1 + 24), (_QWORD *)(*v1 + 8));
  if ( *(_BYTE *)(a1 + 50) )
    v5 = *v1;
  else
    v5 = *v2;
  *(_QWORD *)result = v5;
  *v1 = result;
  *(_BYTE *)(result + 26) = 5;
  *(_QWORD *)(result + 48) = 0;
  *(_BYTE *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = v3;
  return result;
}

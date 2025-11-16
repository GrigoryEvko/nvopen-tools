// Function: sub_396E9A0
// Address: 0x396e9a0
//
__int64 __fastcall sub_396E9A0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // r8
  __int64 v3; // rax

  v1 = *(_QWORD *)(a1 + 256);
  v2 = 0;
  v3 = *(unsigned int *)(v1 + 120);
  if ( (_DWORD)v3 )
    return *(_QWORD *)(*(_QWORD *)(v1 + 112) + 32 * v3 - 32);
  return v2;
}

// Function: sub_C9FC90
// Address: 0xc9fc90
//
__int64 __fastcall sub_C9FC90(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx

  v2 = **(__int64 ***)(__readfsqword(0) - 24);
  v3 = *v2;
  if ( *(_BYTE *)(*v2 + 864) )
  {
    *(_BYTE *)(v3 + 864) = 0;
    sub_C9FAB0(v3 + 840, a2);
  }
  *(_BYTE *)(v3 + 864) = 1;
  *(_QWORD *)(v3 + 840) = 0;
  *(_QWORD *)(v3 + 848) = 0;
  *(_QWORD *)(v3 + 856) = 0x2800000000LL;
  return 0x2800000000LL;
}

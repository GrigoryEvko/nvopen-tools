// Function: sub_15CD6A0
// Address: 0x15cd6a0
//
bool __fastcall sub_15CD6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  _QWORD *v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *i; // rdx

  v2 = *(unsigned int *)(a1 + 308);
  if ( (_DWORD)v2 == *(_DWORD *)(a1 + 312) )
    return 0;
  v3 = *(_QWORD *)(a1 + 296);
  if ( v3 != *(_QWORD *)(a1 + 288) )
    v2 = *(unsigned int *)(a1 + 304);
  v4 = (_QWORD *)(v3 + 8 * v2);
  v5 = sub_15CC2D0(a1 + 280, a2);
  v6 = *(_QWORD *)(a1 + 296);
  if ( v6 == *(_QWORD *)(a1 + 288) )
    v7 = *(unsigned int *)(a1 + 308);
  else
    v7 = *(unsigned int *)(a1 + 304);
  for ( i = (_QWORD *)(v6 + 8 * v7); i != v5; ++v5 )
  {
    if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  return v5 != v4;
}

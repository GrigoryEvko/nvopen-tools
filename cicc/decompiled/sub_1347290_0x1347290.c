// Function: sub_1347290
// Address: 0x1347290
//
bool __fastcall sub_1347290(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rdx

  v1 = sub_134BFA0(a1 + 320);
  if ( !v1 )
    return 0;
  v2 = *(unsigned int *)(a1 + 5640);
  v3 = *(_QWORD *)(a1 + 1368) + 512LL - *(_QWORD *)(a1 + 5664) - *(_QWORD *)(v1 + 176);
  if ( (_DWORD)v2 == -1 )
    return 0;
  v4 = *(_QWORD *)(a1 + 1360);
  if ( v4 <= 0xFFFFFFFFFFFFLL )
    v5 = (v2 * v4) >> 16;
  else
    v5 = v2 * (v4 >> 16);
  return v3 > v5;
}

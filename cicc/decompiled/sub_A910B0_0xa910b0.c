// Function: sub_A910B0
// Address: 0xa910b0
//
__int64 __fastcall sub_A910B0(__int64 a1, __int64 *a2, char a3)
{
  unsigned int v3; // eax
  __int64 v4; // rbx
  unsigned int v5; // r13d
  unsigned int v6; // r12d
  __int64 v8; // rax

  *a2 = 0;
  v3 = sub_A8E250(a1, a2, a3);
  v4 = *a2;
  v5 = v3;
  if ( !*a2 )
    v4 = a1;
  v6 = *(_DWORD *)(v4 + 36);
  if ( v6 )
  {
    v8 = sub_B2BE50(v4);
    *(_QWORD *)(v4 + 120) = sub_B612D0(v8, v6);
  }
  return v5;
}

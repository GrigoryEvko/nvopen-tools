// Function: sub_28CA020
// Address: 0x28ca020
//
__int64 __fastcall sub_28CA020(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  v3 = a2;
  if ( a3 )
  {
    v7 = sub_904010(a2, "etype = ");
    v8 = sub_CB59F0(v7, *(int *)(a1 + 8));
    sub_904010(v8, ",");
  }
  v4 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) <= 8 )
  {
    v3 = sub_CB6200(a2, "opcode = ", 9u);
  }
  else
  {
    *(_BYTE *)(v4 + 8) = 32;
    *(_QWORD *)v4 = 0x3D2065646F63706FLL;
    *(_QWORD *)(a2 + 32) += 9LL;
  }
  v5 = sub_CB59D0(v3, *(unsigned int *)(a1 + 12));
  return sub_904010(v5, ", ");
}

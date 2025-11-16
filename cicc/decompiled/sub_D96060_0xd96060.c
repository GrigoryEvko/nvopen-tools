// Function: sub_D96060
// Address: 0xd96060
//
void __fastcall sub_D96060(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rax

  if ( *(_WORD *)(a2 + 24) )
    goto LABEL_2;
  v2 = sub_D95540(a2);
  sub_A587F0(v2, a1, 0, 0);
  v3 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) != v3 )
  {
    *v3 = 32;
    ++*(_QWORD *)(a1 + 32);
LABEL_2:
    sub_D955C0(a2, a1);
    return;
  }
  sub_CB6200(a1, (unsigned __int8 *)" ", 1u);
  sub_D955C0(a2, a1);
}

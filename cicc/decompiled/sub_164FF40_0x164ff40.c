// Function: sub_164FF40
// Address: 0x164ff40
//
void __fastcall sub_164FF40(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax

  v2 = *a1;
  if ( !*a1 )
    goto LABEL_4;
  sub_16E2CE0(a2, v2);
  v3 = *(_BYTE **)(v2 + 24);
  if ( (unsigned __int64)v3 < *(_QWORD *)(v2 + 16) )
  {
    *(_QWORD *)(v2 + 24) = v3 + 1;
    *v3 = 10;
LABEL_4:
    *((_BYTE *)a1 + 72) = 1;
    return;
  }
  sub_16E7DE0(v2, 10);
  *((_BYTE *)a1 + 72) = 1;
}

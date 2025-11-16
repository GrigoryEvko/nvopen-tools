// Function: sub_BDBF70
// Address: 0xbdbf70
//
void __fastcall sub_BDBF70(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax

  v2 = *a1;
  if ( !*a1 )
    goto LABEL_4;
  sub_CA0E80(a2, v2);
  v3 = *(_BYTE **)(v2 + 32);
  if ( (unsigned __int64)v3 < *(_QWORD *)(v2 + 24) )
  {
    *(_QWORD *)(v2 + 32) = v3 + 1;
    *v3 = 10;
LABEL_4:
    *((_BYTE *)a1 + 152) = 1;
    return;
  }
  sub_CB5D20(v2, 10);
  *((_BYTE *)a1 + 152) = 1;
}

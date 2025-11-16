// Function: sub_BDA210
// Address: 0xbda210
//
void __fastcall sub_BDA210(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  _BYTE *v4; // rax

  v2 = *a1;
  v3 = **a1;
  if ( !v3 )
    goto LABEL_4;
  sub_CA0E80(a2, v3);
  v4 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v4 < *(_QWORD *)(v3 + 24) )
  {
    *(_QWORD *)(v3 + 32) = v4 + 1;
    *v4 = 10;
LABEL_4:
    *((_BYTE *)v2 + 152) = 1;
    return;
  }
  sub_CB5D20(v3, 10);
  *((_BYTE *)v2 + 152) = 1;
}

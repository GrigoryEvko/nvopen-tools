// Function: sub_869FD0
// Address: 0x869fd0
//
__int64 __fastcall sub_869FD0(_QWORD *a1, int a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rsi

  v2 = (_QWORD *)a1[1];
  v3 = *a1;
  if ( a2 == -1 )
  {
    if ( v2 )
      *v2 = v3;
    else
      *(_QWORD *)(qword_4F07288 + 256) = v3;
    v4 = *a1;
    if ( !*a1 )
      goto LABEL_7;
    v5 = a1[1];
LABEL_6:
    *(_QWORD *)(v4 + 8) = v5;
    *a1 = 0;
LABEL_7:
    a1[1] = 0;
    return sub_869F90(a1);
  }
  v7 = qword_4F04C68[0] + 776LL * a2;
  if ( v2 )
    *v2 = v3;
  else
    *(_QWORD *)(v7 + 328) = v3;
  v4 = *a1;
  v5 = a1[1];
  if ( *a1 )
    goto LABEL_6;
  *(_QWORD *)(v7 + 336) = v5;
  a1[1] = 0;
  return sub_869F90(a1);
}

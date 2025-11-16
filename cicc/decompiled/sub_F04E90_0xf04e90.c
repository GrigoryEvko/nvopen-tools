// Function: sub_F04E90
// Address: 0xf04e90
//
__int64 __fastcall sub_F04E90(__int64 a1, char *a2)
{
  __int64 v2; // r12
  _BYTE *v4; // rax
  unsigned int v5; // r13d
  _BYTE *v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  _BYTE *v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rdi

  v2 = a1;
  sub_CB59D0(a1, *(unsigned int *)a2);
  if ( a2[7] >= 0 )
  {
    if ( a2[11] >= 0 )
      goto LABEL_3;
LABEL_8:
    v6 = *(_BYTE **)(v2 + 32);
    v7 = *((_DWORD *)a2 + 2) & 0x7FFFFFFF;
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v2 + 24) )
    {
      v8 = sub_CB5D20(v2, 46);
    }
    else
    {
      v8 = v2;
      *(_QWORD *)(v2 + 32) = v6 + 1;
      *v6 = 46;
    }
    sub_CB59D0(v8, v7);
    if ( a2[15] >= 0 )
      return v2;
    goto LABEL_11;
  }
  v4 = *(_BYTE **)(a1 + 32);
  v5 = *((_DWORD *)a2 + 1) & 0x7FFFFFFF;
  if ( (unsigned __int64)v4 >= *(_QWORD *)(a1 + 24) )
  {
    a1 = sub_CB5D20(a1, 46);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v4 + 1;
    *v4 = 46;
  }
  sub_CB59D0(a1, v5);
  if ( a2[11] < 0 )
    goto LABEL_8;
LABEL_3:
  if ( a2[15] >= 0 )
    return v2;
LABEL_11:
  v9 = *(_BYTE **)(v2 + 32);
  v10 = *((_DWORD *)a2 + 3) & 0x7FFFFFFF;
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v2 + 24) )
  {
    v11 = sub_CB5D20(v2, 46);
  }
  else
  {
    v11 = v2;
    *(_QWORD *)(v2 + 32) = v9 + 1;
    *v9 = 46;
  }
  sub_CB59D0(v11, v10);
  return v2;
}

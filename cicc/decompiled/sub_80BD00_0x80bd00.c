// Function: sub_80BD00
// Address: 0x80bd00
//
__int64 __fastcall sub_80BD00(_QWORD *a1, __int64 a2)
{
  char *v2; // r13
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _QWORD *v6; // rax

  if ( dword_4F18BD8 )
  {
    *(_DWORD *)(a2 + 48) = 1;
    v2 = (char *)byte_3F871B3;
  }
  else
  {
    if ( *a1 )
      v6 = sub_72B7A0(a1);
    else
      v6 = (_QWORD *)qword_4D03FF0;
    v2 = *(char **)v6[49];
    if ( !v2 )
      v2 = sub_723F40(0);
  }
  v3 = (_QWORD *)qword_4F18BE0;
  ++*(_QWORD *)a2;
  v4 = v3[2];
  if ( (unsigned __int64)(v4 + 1) > v3[1] )
  {
    sub_823810(v3);
    v3 = (_QWORD *)qword_4F18BE0;
    v4 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v3[4] + v4) = 66;
  ++v3[2];
  return sub_80BC40(v2, (_QWORD *)a2);
}

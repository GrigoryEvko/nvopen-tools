// Function: sub_1441BF0
// Address: 0x1441bf0
//
char __fastcall sub_1441BF0(__int64 a1)
{
  char result; // al
  unsigned int **v2; // r13
  unsigned int *v3; // rax
  bool v4; // zf
  unsigned int *v5; // r12
  unsigned int *v6; // rax

  result = sub_1441AE0((_QWORD *)a1);
  if ( result )
  {
    v2 = (unsigned int **)(*(_QWORD *)(a1 + 8) + 8LL);
    v3 = sub_1441840(v2, dword_4F9A020);
    v4 = *(_BYTE *)(a1 + 24) == 0;
    v5 = v3;
    *(_QWORD *)(a1 + 16) = *((_QWORD *)v3 + 1);
    if ( v4 )
      *(_BYTE *)(a1 + 24) = 1;
    v6 = sub_1441840(v2, dword_4F99F40);
    v4 = *(_BYTE *)(a1 + 40) == 0;
    *(_QWORD *)(a1 + 32) = *((_QWORD *)v6 + 1);
    if ( v4 )
      *(_BYTE *)(a1 + 40) = 1;
    result = *((_QWORD *)v5 + 2) > (unsigned __int64)(unsigned int)dword_4F99D80;
    v4 = *(_BYTE *)(a1 + 49) == 0;
    *(_BYTE *)(a1 + 48) = result;
    if ( v4 )
      *(_BYTE *)(a1 + 49) = 1;
  }
  return result;
}

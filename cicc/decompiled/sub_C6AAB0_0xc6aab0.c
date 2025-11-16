// Function: sub_C6AAB0
// Address: 0xc6aab0
//
__int64 __fastcall sub_C6AAB0(__int64 a1)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rdi
  _BYTE *v5; // rax

  v2 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8;
  if ( *(_BYTE *)(v2 + 4) )
  {
    v4 = *(_QWORD *)(a1 + 160);
    v5 = *(_BYTE **)(v4 + 32);
    if ( (unsigned __int64)v5 < *(_QWORD *)(v4 + 24) )
    {
      *(_QWORD *)(v4 + 32) = v5 + 1;
      *v5 = 44;
      if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) != 1 )
        goto LABEL_3;
LABEL_6:
      sub_C6A6A0(a1);
      goto LABEL_3;
    }
    sub_CB5D20(v4, 44);
    v2 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8;
  }
  if ( *(_DWORD *)v2 == 1 )
    goto LABEL_6;
LABEL_3:
  sub_C6A6F0(a1);
  result = *(_QWORD *)a1;
  *(_BYTE *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 4) = 1;
  return result;
}

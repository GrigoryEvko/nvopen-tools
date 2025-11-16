// Function: sub_80C040
// Address: 0x80c040
//
void __fastcall sub_80C040(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rbx
  char v3; // al
  unsigned __int64 v4; // rdi
  __int64 v5; // rax

  v2 = *a1;
  if ( *a1 )
  {
    v3 = *(_BYTE *)(v2 + 80);
    if ( (*((_BYTE *)a1 + 89) & 1) != 0 )
    {
      if ( v3 == 2 )
      {
        if ( !sub_72AE00(*(_QWORD *)(v2 + 88)) )
          return;
        v4 = *(unsigned int *)(v2 + 44);
        goto LABEL_14;
      }
      if ( v3 != 7 )
      {
        if ( (unsigned __int8)(v3 - 4) <= 1u )
        {
          v5 = *(_QWORD *)(v2 + 96);
          if ( !v5 )
            return;
          v4 = *(_QWORD *)(v5 + 168);
        }
        else if ( v3 == 6 )
        {
          v4 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 16LL);
        }
        else
        {
          if ( v3 != 3 )
            return;
          v4 = *(_QWORD *)(v2 + 96);
        }
        goto LABEL_14;
      }
LABEL_11:
      v4 = *(_QWORD *)(v2 + 104);
LABEL_14:
      if ( v4 > 1 )
        sub_80BEC0(v4, 1, a2);
      return;
    }
    if ( v3 == 7 && *(char *)(*(_QWORD *)(v2 + 88) + 173LL) < 0 )
      goto LABEL_11;
  }
}

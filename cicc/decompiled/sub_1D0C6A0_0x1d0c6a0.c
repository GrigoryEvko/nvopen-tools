// Function: sub_1D0C6A0
// Address: 0x1d0c6a0
//
void __fastcall sub_1D0C6A0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  int v6; // eax
  unsigned int *v7; // rax
  __int16 v8; // ax
  __int64 v9; // rdi
  __int64 (*v10)(); // rdx

  v2 = *a2;
  if ( *a2 && *(_WORD *)(v2 + 24) == 2 )
  {
    *((_WORD *)a2 + 113) = 0;
    return;
  }
  v3 = *(__int64 (**)())(*a1 + 104LL);
  if ( v3 != sub_1CFBF50 && ((unsigned __int8 (__fastcall *)(_QWORD *))v3)(a1) )
    goto LABEL_15;
  v4 = a1[79];
  if ( !v4 || !*(_QWORD *)(v4 + 96) )
  {
    if ( v2 )
    {
      v8 = *(_WORD *)(v2 + 24);
      if ( v8 < 0 )
      {
        v9 = a1[2];
        v10 = *(__int64 (**)())(*(_QWORD *)v9 + 872LL);
        if ( v10 != sub_1D0B180 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v10)(v9, (unsigned int)~v8) )
          {
            *((_WORD *)a2 + 113) = dword_4FC1580;
            return;
          }
        }
      }
    }
LABEL_15:
    *((_WORD *)a2 + 113) = 1;
    return;
  }
  v5 = *a2;
  *((_WORD *)a2 + 113) = 0;
  do
  {
    if ( !v5 )
      break;
    if ( *(__int16 *)(v5 + 24) < 0 )
      *((_WORD *)a2 + 113) += (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a1[2] + 864LL))(
                                a1[2],
                                a1[79],
                                v5);
    v6 = *(_DWORD *)(v5 + 56);
    if ( !v6 )
      break;
    v7 = (unsigned int *)(*(_QWORD *)(v5 + 32) + 40LL * (unsigned int)(v6 - 1));
    v5 = *(_QWORD *)v7;
  }
  while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2]) == 111 );
}

// Function: sub_335CEF0
// Address: 0x335cef0
//
void __fastcall sub_335CEF0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 (*v4)(); // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // eax
  unsigned int *v8; // rax
  int v9; // esi
  __int64 v10; // rdi
  __int64 (*v11)(); // rax

  v3 = *a2;
  if ( *a2 && *(_DWORD *)(v3 + 24) == 2 )
  {
    *((_WORD *)a2 + 126) = 0;
    return;
  }
  v4 = *(__int64 (**)())(*a1 + 112LL);
  if ( v4 != sub_334CAA0 && ((unsigned __int8 (__fastcall *)(_QWORD *))v4)(a1) )
    goto LABEL_15;
  v5 = a1[75];
  if ( !v5 || !*(_QWORD *)(v5 + 104) )
  {
    if ( v3 )
    {
      v9 = *(_DWORD *)(v3 + 24);
      if ( v9 < 0 )
      {
        v10 = a1[2];
        v11 = *(__int64 (**)())(*(_QWORD *)v10 + 1184LL);
        if ( v11 != sub_2FDC750 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v11)(v10, (unsigned int)~v9) )
          {
            *((_WORD *)a2 + 126) = qword_50390E8;
            return;
          }
        }
      }
    }
LABEL_15:
    *((_WORD *)a2 + 126) = 1;
    return;
  }
  v6 = *a2;
  *((_WORD *)a2 + 126) = 0;
  do
  {
    if ( !v6 )
      break;
    if ( *(int *)(v6 + 24) < 0 )
      *((_WORD *)a2 + 126) += (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a1[2] + 1176LL))(
                                a1[2],
                                a1[75],
                                v6);
    v7 = *(_DWORD *)(v6 + 64);
    if ( !v7 )
      break;
    v8 = (unsigned int *)(*(_QWORD *)(v6 + 40) + 40LL * (unsigned int)(v7 - 1));
    v6 = *(_QWORD *)v8;
  }
  while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v8[2]) == 262 );
}

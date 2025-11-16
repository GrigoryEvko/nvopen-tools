// Function: sub_335E3B0
// Address: 0x335e3b0
//
void __fastcall sub_335E3B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int16 v9; // ax
  int v10; // eax
  unsigned int *v11; // rax

  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(unsigned int *)(a1 + 16);
      if ( *(_DWORD *)(a1 + 20) > (unsigned int)v8 )
        break;
LABEL_7:
      v10 = *(_DWORD *)(v7 + 64);
      if ( !v10
        || (v11 = (unsigned int *)(*(_QWORD *)(v7 + 40) + 40LL * (unsigned int)(v10 - 1)),
            *(_WORD *)(*(_QWORD *)(*(_QWORD *)v11 + 48LL) + 16LL * v11[2]) != 262) )
      {
        *(_QWORD *)(a1 + 8) = 0;
        return;
      }
      *(_QWORD *)(a1 + 8) = *(_QWORD *)v11;
      sub_335E330(a1);
      v7 = *(_QWORD *)(a1 + 8);
      if ( !v7 )
        return;
    }
    while ( !(unsigned __int8)sub_33CF8A0(v7, v8, a3, a4, a5, a6) )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
      *(_DWORD *)(a1 + 16) = v8;
      if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v8 )
        goto LABEL_7;
    }
    v9 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL) + 16LL * (unsigned int)(*(_DWORD *)(a1 + 16))++);
    *(_WORD *)(a1 + 24) = v9;
  }
}

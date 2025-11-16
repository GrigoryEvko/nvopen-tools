// Function: sub_30FCF60
// Address: 0x30fcf60
//
void __fastcall sub_30FCF60(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, char a5)
{
  bool v8; // zf
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r12
  unsigned __int64 v17; // rdi

  sub_30CABE0(a1, a2, a3, a4, a5);
  v8 = *(_BYTE *)(a2 + 360) == 0;
  *(_QWORD *)a1 = &unk_4A328D0;
  if ( !v8 )
  {
    *(_QWORD *)(a1 + 64) = 0;
LABEL_3:
    *(_QWORD *)(a1 + 72) = 0;
    v9 = 0;
    goto LABEL_4;
  }
  v10 = sub_30FCBF0(a2, *(_QWORD *)(a1 + 16));
  v8 = *(_BYTE *)(a2 + 360) == 0;
  *(_QWORD *)(a1 + 64) = v10[8];
  if ( !v8 )
    goto LABEL_3;
  v11 = sub_30FCBF0(a2, *(_QWORD *)(a1 + 24));
  v8 = *(_BYTE *)(a2 + 360) == 0;
  *(_QWORD *)(a1 + 72) = v11[8];
  if ( v8 )
  {
    v12 = sub_30FCC90(a2, *(_QWORD *)(a1 + 16));
    v9 = v12 + sub_30FCC90(a2, *(_QWORD *)(a1 + 24));
  }
  else
  {
    v9 = 0;
  }
LABEL_4:
  *(_QWORD *)(a1 + 80) = v9;
  qmemcpy((void *)(a1 + 88), sub_30FCBF0(a2, *(_QWORD *)(a1 + 16)), 0x160u);
  *(_BYTE *)(a1 + 544) = 0;
  if ( a5 )
  {
    v16 = sub_30FCBF0(a2, *(_QWORD *)(a1 + 16));
    if ( *(_BYTE *)(a1 + 544) )
    {
      v17 = *(_QWORD *)(a1 + 496);
      *(_BYTE *)(a1 + 544) = 0;
      if ( v17 != a1 + 512 )
        _libc_free(v17);
      sub_C7D6A0(*(_QWORD *)(a1 + 472), 8LL * *(unsigned int *)(a1 + 488), 8);
    }
    sub_30C62E0(a1 + 440, v16, (__int64)a3, v13, v14, v15);
    *(_BYTE *)(a1 + 544) = 1;
  }
}

// Function: sub_1F392E0
// Address: 0x1f392e0
//
__int64 __fastcall sub_1F392E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 *v9; // r15
  char v10; // r12
  __int64 *v11; // r13
  __int64 v13; // r13
  __int64 v14; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned __int8 *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 40);
  if ( !(_BYTE)v7 || (v7 = (unsigned __int8)byte_4FCAFE0, !byte_4FCAFE0) )
  {
    v9 = *(__int64 **)(*(_QWORD *)(v8 + 328) + 8LL);
    v14 = v8 + 320;
    if ( v9 == (__int64 *)(v8 + 320) )
      return v7;
    goto LABEL_4;
  }
  sub_1F38140(v8, 1, a3, a4, a5, a6);
  v13 = *(_QWORD *)(a1 + 40);
  v9 = *(__int64 **)(*(_QWORD *)(v13 + 328) + 8LL);
  v14 = v13 + 320;
  if ( v9 != (__int64 *)(v13 + 320) )
  {
LABEL_4:
    v7 = 0;
    do
    {
      v11 = v9;
      v9 = (__int64 *)v9[1];
      if ( dword_4FCAF00 == dword_4CD4AD8 )
        break;
      v10 = sub_1F34100(v11);
      if ( (unsigned __int8)sub_1F34340(a1, v10, v11) )
        v7 |= sub_1F389B0(a1, v10, (__int64)v11, 0, 0, 0);
    }
    while ( v9 != (__int64 *)v14 );
    goto LABEL_9;
  }
  v7 = 0;
LABEL_9:
  if ( *(_BYTE *)(a1 + 48) && byte_4FCAFE0 )
    sub_1F38140(*(_QWORD *)(a1 + 40), 0, a3, a4, a5, a6);
  return v7;
}

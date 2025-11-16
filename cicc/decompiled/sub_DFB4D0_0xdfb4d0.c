// Function: sub_DFB4D0
// Address: 0xdfb4d0
//
__int64 __fastcall sub_DFB4D0(_QWORD *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // r8
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rdx
  __int64 (*v15)(); // rax
  __int64 v16; // [rsp+8h] [rbp-38h]
  unsigned int v17; // [rsp+10h] [rbp-30h]

  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_19;
  v11 = v3[7];
  v10 = v3 + 6;
  if ( !v11 )
    goto LABEL_19;
  v4 = (unsigned int)dword_4F89D28;
  v12 = v3 + 6;
  do
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v11 + 16);
      v13 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_4F89D28 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v13 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v9 );
LABEL_15:
  if ( v10 == v12 || dword_4F89D28 < *((_DWORD *)v12 + 8) || *((int *)v12 + 9) <= 0 )
  {
LABEL_19:
    v15 = *(__int64 (**)())(*(_QWORD *)*a1 + 1112LL);
    if ( v15 == sub_DF60B0 )
      return v17;
    else
      return ((__int64 (__fastcall *)(_QWORD, unsigned __int64, __int64 (*)(), __int64, _QWORD *))v15)(
               *a1,
               v4,
               sub_DF60B0,
               v9,
               v10);
  }
  else
  {
    BYTE4(v16) = 1;
    LODWORD(v16) = qword_4F89DA8;
    return v16;
  }
}

// Function: sub_DFB3D0
// Address: 0xdfb3d0
//
__int64 __fastcall sub_DFB3D0(_QWORD *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // r8
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 (*v15)(); // rdx

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
  v10 = v3[7];
  v9 = v3 + 6;
  if ( !v10 )
    goto LABEL_19;
  v4 = (unsigned int)dword_4F89E08;
  v11 = v3 + 6;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v10 + 16);
      v13 = *(_QWORD *)(v10 + 24);
      if ( *(_DWORD *)(v10 + 32) >= dword_4F89E08 )
        break;
      v10 = *(_QWORD *)(v10 + 24);
      if ( !v13 )
        goto LABEL_15;
    }
    v11 = (_QWORD *)v10;
    v10 = *(_QWORD *)(v10 + 16);
  }
  while ( v12 );
LABEL_15:
  if ( v9 == v11
    || dword_4F89E08 < *((_DWORD *)v11 + 8)
    || (result = (unsigned int)qword_4F89E88, *((int *)v11 + 9) <= 0) )
  {
LABEL_19:
    v15 = *(__int64 (**)())(*(_QWORD *)*a1 + 1088LL);
    result = 0;
    if ( v15 != sub_DF60A0 )
      return ((__int64 (__fastcall *)(_QWORD, unsigned __int64, __int64 (*)(), __int64 (*)(), _QWORD *))v15)(
               *a1,
               v4,
               v15,
               sub_DF60A0,
               v9);
  }
  return result;
}

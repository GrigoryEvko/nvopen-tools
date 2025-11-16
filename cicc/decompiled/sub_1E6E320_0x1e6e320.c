// Function: sub_1E6E320
// Address: 0x1e6e320
//
_QWORD *__fastcall sub_1E6E320(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 (*v10)(); // rax
  __int64 *v11; // r15
  unsigned int i; // r12d
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 (__fastcall *v15)(__int64, unsigned __int8); // rax
  _DWORD *v16; // rax
  __int64 v17; // rdi
  void (*v18)(); // rax
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  _DWORD *v25; // r8
  _DWORD *v26; // rdi
  int v27; // esi
  __int64 v28; // rcx
  __int64 v29; // rdx
  _DWORD *v30; // rdi
  unsigned __int64 v31; // rsi
  _QWORD *result; // rax
  __int64 v33; // rcx
  __int64 v34; // rdx
  _DWORD *v35; // r8
  _DWORD *v36; // rdi
  int v37; // esi
  __int64 v38; // rcx
  __int64 v39; // rdx
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-40h]
  __int64 v43; // [rsp+0h] [rbp-40h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]

  v6 = sub_1E15F70(a2);
  v7 = 0;
  v8 = *(_QWORD *)(v6 + 16);
  v9 = v6;
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 56LL);
  if ( v10 != sub_1D12D20 )
    v7 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v10)(v8, a2, 0);
  *(_BYTE *)(a1 + 136) = 1;
  v11 = (__int64 *)(v7 + 160);
  for ( i = 5; i != 2; --i )
  {
    v13 = *v11;
    if ( *v11 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
      v15 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v7 + 288LL);
      if ( v15 != sub_1D45FB0 )
      {
        v43 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
        v45 = v7;
        v41 = v15(v7, i);
        v14 = v43;
        v7 = v45;
        v13 = v41;
      }
      v16 = (_DWORD *)(*(_QWORD *)v14 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL));
      if ( *(_DWORD *)(v14 + 8) != *v16 )
      {
        v42 = *(_QWORD *)v14 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL);
        v44 = v7;
        sub_1ED7890(v14);
        v16 = (_DWORD *)v42;
        v7 = v44;
      }
      *(_BYTE *)(a1 + 136) = v16[1] >> 1 < a4;
    }
    --v11;
  }
  *(_BYTE *)(a1 + 139) = 1;
  v17 = *(_QWORD *)(v9 + 16);
  v18 = *(void (**)())(*(_QWORD *)v17 + 200LL);
  if ( v18 != nullsub_708 )
    ((void (__fastcall *)(__int64, __int64, _QWORD))v18)(v17, a1 + 136, a4);
  if ( !byte_4FC7BC0 )
    *(_BYTE *)(a1 + 136) = 0;
  v19 = sub_16D5D50();
  v20 = *(_QWORD **)&dword_4FA0208[2];
  v21 = dword_4FA0208;
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    do
    {
      while ( 1 )
      {
        v22 = v20[2];
        v23 = v20[3];
        if ( v19 <= v20[4] )
          break;
        v20 = (_QWORD *)v20[3];
        if ( !v23 )
          goto LABEL_20;
      }
      v21 = v20;
      v20 = (_QWORD *)v20[2];
    }
    while ( v22 );
LABEL_20:
    if ( v21 != dword_4FA0208 && v19 >= *((_QWORD *)v21 + 4) )
    {
      v24 = *((_QWORD *)v21 + 7);
      v25 = v21 + 12;
      if ( v24 )
      {
        v26 = v21 + 12;
        v27 = qword_4FC7DC0[1];
        do
        {
          while ( 1 )
          {
            v28 = *(_QWORD *)(v24 + 16);
            v29 = *(_QWORD *)(v24 + 24);
            if ( *(_DWORD *)(v24 + 32) >= v27 )
              break;
            v24 = *(_QWORD *)(v24 + 24);
            if ( !v29 )
              goto LABEL_27;
          }
          v26 = (_DWORD *)v24;
          v24 = *(_QWORD *)(v24 + 16);
        }
        while ( v28 );
LABEL_27:
        if ( v25 != v26 && v27 >= v26[8] && (int)v26[9] > 0 )
        {
          v40 = qword_4FC7DC0[20];
          *(_BYTE *)(a1 + 139) = v40;
          if ( v40 )
            *(_BYTE *)(a1 + 138) = 0;
        }
      }
    }
  }
  v30 = dword_4FA0208;
  v31 = sub_16D5D50();
  result = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    do
    {
      while ( 1 )
      {
        v33 = result[2];
        v34 = result[3];
        if ( v31 <= result[4] )
          break;
        result = (_QWORD *)result[3];
        if ( !v34 )
          goto LABEL_34;
      }
      v30 = result;
      result = (_QWORD *)result[2];
    }
    while ( v33 );
LABEL_34:
    result = dword_4FA0208;
    if ( v30 != dword_4FA0208 && v31 >= *((_QWORD *)v30 + 4) )
    {
      result = (_QWORD *)*((_QWORD *)v30 + 7);
      v35 = v30 + 12;
      if ( result )
      {
        v36 = v30 + 12;
        v37 = qword_4FC7EA0[1];
        do
        {
          while ( 1 )
          {
            v38 = result[2];
            v39 = result[3];
            if ( *((_DWORD *)result + 8) >= v37 )
              break;
            result = (_QWORD *)result[3];
            if ( !v39 )
              goto LABEL_41;
          }
          v36 = result;
          result = (_QWORD *)result[2];
        }
        while ( v38 );
LABEL_41:
        if ( v35 != v36 && v37 >= v36[8] && (int)v36[9] > 0 )
        {
          result = (_QWORD *)LOBYTE(qword_4FC7EA0[20]);
          *(_BYTE *)(a1 + 138) = (_BYTE)result;
          if ( (_BYTE)result )
            *(_BYTE *)(a1 + 139) = 0;
        }
      }
    }
  }
  return result;
}

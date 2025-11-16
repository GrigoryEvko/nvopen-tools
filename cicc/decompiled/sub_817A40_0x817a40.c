// Function: sub_817A40
// Address: 0x817a40
//
void __fastcall sub_817A40(__int64 a1, __int64 a2, int a3, __int64 *a4)
{
  __int64 i; // r14
  __int64 v8; // rax
  char v9; // al
  unsigned __int8 v10; // al
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r14
  char *v14; // rsi
  _QWORD *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  char *v21; // rsi
  __int64 v22; // [rsp-48h] [rbp-48h] BYREF
  const __m128i *v23; // [rsp-40h] [rbp-40h] BYREF

  if ( !a1 )
    BUG();
  for ( i = a1; ; i = v8 )
  {
    v8 = sub_730290(i);
    if ( (*(_BYTE *)(v8 + 51) & 0x40) != 0 )
      break;
    if ( i == v8 )
      goto LABEL_7;
LABEL_4:
    ;
  }
  v8 = sub_730770(v8, 0);
  if ( i != v8 )
    goto LABEL_4;
LABEL_7:
  v9 = *(_BYTE *)(i + 50);
  if ( (v9 & 0x40) != 0 )
  {
    sub_809D10(i, &v22, (__int64 *)&v23);
    sub_817960(v22, v23, a2, a4);
    return;
  }
  if ( (v9 & 0x10) != 0 )
  {
    v11 = sub_6E3F50(i);
    v12 = v11;
    if ( v11 )
    {
      v13 = 0;
      do
      {
        if ( (*(_BYTE *)(v11 + 25) & 0x10) != 0 )
          break;
        v11 = *(_QWORD *)(v11 + 16);
        ++v13;
      }
      while ( v11 );
      v14 = "cv";
      if ( a3 && !dword_4D0425C )
        v14 = "sc";
      *a4 += 2;
      sub_8238B0(qword_4F18BE0, v14, 2);
      sub_80F5E0(a2, 0, a4);
      if ( v13 == 1 )
      {
        sub_817850(v12, 1u, a4);
        return;
      }
    }
    else
    {
      if ( !a3 || dword_4D0425C )
      {
        *a4 += 2;
        v21 = "cv";
      }
      else
      {
        *a4 += 2;
        v21 = "sc";
      }
      sub_8238B0(qword_4F18BE0, v21, 2);
      sub_80F5E0(a2, 0, a4);
    }
    v15 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v16 = v15[2];
    if ( (unsigned __int64)(v16 + 1) > v15[1] )
    {
      sub_823810(v15);
      v15 = (_QWORD *)qword_4F18BE0;
      v16 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v15[4] + v16) = 95;
    ++v15[2];
    sub_817850(v12, 1u, a4);
    v17 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v18 = v17[2];
    if ( (unsigned __int64)(v18 + 1) > v17[1] )
    {
      sub_823810(v17);
      v17 = (_QWORD *)qword_4F18BE0;
      v18 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v17[4] + v18) = 69;
    ++v17[2];
  }
  else
  {
    v10 = *(_BYTE *)(i + 48);
    if ( v10 == 3 )
    {
      v20 = sub_6E3F50(i);
      sub_816460(v20, 1u, 0, a4);
    }
    else if ( v10 > 3u )
    {
      if ( v10 != 5 )
        goto LABEL_40;
      v19 = sub_6E3F50(i);
      if ( v19 )
        sub_817850(v19, 1u, a4);
    }
    else if ( v10 != 1 )
    {
      if ( v10 == 2 )
      {
        sub_80D8A0(*(const __m128i **)(i + 56), 1u, 0, a4);
        return;
      }
LABEL_40:
      sub_721090();
    }
  }
}

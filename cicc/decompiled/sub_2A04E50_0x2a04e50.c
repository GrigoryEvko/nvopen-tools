// Function: sub_2A04E50
// Address: 0x2a04e50
//
__int64 __fastcall sub_2A04E50(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 a4, unsigned __int16 a5, char a6)
{
  _QWORD *v10; // r14
  _QWORD *v11; // r13
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // r14
  _QWORD *v22; // r13
  unsigned __int64 v23; // rsi
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  _QWORD *v32; // r14
  _QWORD *v33; // r13
  unsigned __int64 v34; // rsi
  _QWORD *v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // [rsp+8h] [rbp-28h]

  LODWORD(v43) = 0;
  WORD2(v43) = 1;
  BYTE6(v43) = 1;
  sub_DFA090(a3);
  if ( a6 )
  {
    v10 = sub_C52410();
    v11 = v10 + 1;
    v12 = sub_C959E0();
    v13 = (_QWORD *)v10[2];
    if ( v13 )
    {
      v14 = v10 + 1;
      do
      {
        while ( 1 )
        {
          v15 = v13[2];
          v16 = v13[3];
          if ( v12 <= v13[4] )
            break;
          v13 = (_QWORD *)v13[3];
          if ( !v16 )
            goto LABEL_12;
        }
        v14 = v13;
        v13 = (_QWORD *)v13[2];
      }
      while ( v15 );
LABEL_12:
      if ( v11 != v14 && v12 >= v14[4] )
        v11 = v14;
    }
    if ( v11 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v17 = v11[7];
      if ( v17 )
      {
        v18 = v11 + 6;
        do
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v17 + 16);
            v20 = *(_QWORD *)(v17 + 24);
            if ( *(_DWORD *)(v17 + 32) >= dword_5009E28 )
              break;
            v17 = *(_QWORD *)(v17 + 24);
            if ( !v20 )
              goto LABEL_21;
          }
          v18 = (_QWORD *)v17;
          v17 = *(_QWORD *)(v17 + 16);
        }
        while ( v19 );
LABEL_21:
        if ( v11 + 6 != v18 && dword_5009E28 >= *((_DWORD *)v18 + 8) && *((int *)v18 + 9) > 0 )
          LODWORD(v43) = qword_5009EA8;
      }
    }
    v21 = sub_C52410();
    v22 = v21 + 1;
    v23 = sub_C959E0();
    v24 = (_QWORD *)v21[2];
    if ( v24 )
    {
      v25 = v21 + 1;
      do
      {
        while ( 1 )
        {
          v26 = v24[2];
          v27 = v24[3];
          if ( v23 <= v24[4] )
            break;
          v24 = (_QWORD *)v24[3];
          if ( !v27 )
            goto LABEL_30;
        }
        v25 = v24;
        v24 = (_QWORD *)v24[2];
      }
      while ( v26 );
LABEL_30:
      if ( v22 != v25 && v23 >= v25[4] )
        v22 = v25;
    }
    if ( v22 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v28 = v22[7];
      if ( v28 )
      {
        v29 = v22 + 6;
        do
        {
          while ( 1 )
          {
            v30 = *(_QWORD *)(v28 + 16);
            v31 = *(_QWORD *)(v28 + 24);
            if ( *(_DWORD *)(v28 + 32) >= dword_5009D48 )
              break;
            v28 = *(_QWORD *)(v28 + 24);
            if ( !v31 )
              goto LABEL_39;
          }
          v29 = (_QWORD *)v28;
          v28 = *(_QWORD *)(v28 + 16);
        }
        while ( v30 );
LABEL_39:
        if ( v22 + 6 != v29 && dword_5009D48 >= *((_DWORD *)v29 + 8) && *((int *)v29 + 9) > 0 )
          BYTE4(v43) = qword_5009DC8;
      }
    }
    v32 = sub_C52410();
    v33 = v32 + 1;
    v34 = sub_C959E0();
    v35 = (_QWORD *)v32[2];
    if ( v35 )
    {
      v36 = v32 + 1;
      do
      {
        while ( 1 )
        {
          v37 = v35[2];
          v38 = v35[3];
          if ( v34 <= v35[4] )
            break;
          v35 = (_QWORD *)v35[3];
          if ( !v38 )
            goto LABEL_48;
        }
        v36 = v35;
        v35 = (_QWORD *)v35[2];
      }
      while ( v37 );
LABEL_48:
      if ( v33 != v36 && v34 >= v36[4] )
        v33 = v36;
    }
    if ( v33 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v39 = v33[7];
      if ( v39 )
      {
        v40 = v33 + 6;
        do
        {
          while ( 1 )
          {
            v41 = *(_QWORD *)(v39 + 16);
            v42 = *(_QWORD *)(v39 + 24);
            if ( *(_DWORD *)(v39 + 32) >= dword_5009C68 )
              break;
            v39 = *(_QWORD *)(v39 + 24);
            if ( !v42 )
              goto LABEL_57;
          }
          v40 = (_QWORD *)v39;
          v39 = *(_QWORD *)(v39 + 16);
        }
        while ( v41 );
LABEL_57:
        if ( v33 + 6 != v40 && dword_5009C68 >= *((_DWORD *)v40 + 8) && *((int *)v40 + 9) > 0 )
          BYTE5(v43) = qword_5009CE8;
      }
    }
  }
  if ( HIBYTE(a4) )
    BYTE4(v43) = a4;
  if ( HIBYTE(a5) )
    BYTE6(v43) = a5;
  return v43;
}

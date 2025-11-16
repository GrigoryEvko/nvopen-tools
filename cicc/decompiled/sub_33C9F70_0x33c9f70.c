// Function: sub_33C9F70
// Address: 0x33c9f70
//
_QWORD *__fastcall sub_33C9F70(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 *v8; // rax
  char v9; // al
  _QWORD *v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // rsi
  _QWORD *v13; // rdi
  int v15; // eax
  __int64 v16; // rsi
  char v17; // si
  char v18; // di
  int v19; // eax
  int v20; // eax

  v3 = a2;
  v6 = (_QWORD *)*a2;
  v7 = sub_C33340();
  if ( v6 == v7 )
    v8 = (__int64 *)a2[1];
  else
    v8 = a2;
  v9 = *((_BYTE *)v8 + 20);
  v10 = *(_QWORD **)a3;
  if ( (v9 & 7) != 1 )
  {
    v16 = a3;
    if ( v7 == v10 )
      v16 = *(_QWORD *)(a3 + 8);
    v17 = *(_BYTE *)(v16 + 20);
    v18 = v17 & 7;
    if ( (v17 & 7) == 1 )
    {
      v12 = v3;
      v13 = a1;
      if ( v6 != v7 )
        goto LABEL_20;
LABEL_12:
      sub_C3C790(v13, (_QWORD **)v12);
      return a1;
    }
    if ( (v9 & 7) == 3 )
    {
      if ( v7 != v10 )
      {
        if ( v18 == 3 )
        {
          v19 = v9 & 8;
          if ( (v19 != 0) != ((v17 & 8) != 0) )
          {
            if ( !(_BYTE)v19 )
            {
              v3 = (__int64 *)a3;
LABEL_30:
              sub_C33EB0(a1, v3);
              return a1;
            }
LABEL_35:
            if ( v6 != v7 )
              goto LABEL_30;
            a3 = (__int64)v3;
LABEL_11:
            v12 = (__int64 *)a3;
            v13 = a1;
            goto LABEL_12;
          }
        }
        goto LABEL_15;
      }
      if ( v18 == 3 )
      {
        v20 = v9 & 8;
        if ( ((v17 & 8) != 0) != (v20 != 0) )
        {
          if ( !(_BYTE)v20 )
            goto LABEL_11;
          goto LABEL_35;
        }
      }
    }
    else if ( v7 != v10 )
    {
LABEL_15:
      v15 = sub_C37950(a3, (__int64)v3);
      goto LABEL_16;
    }
    v15 = sub_C3E510(a3, (__int64)v3);
LABEL_16:
    if ( v15 )
      a3 = (__int64)v3;
    if ( v7 != *(_QWORD **)a3 )
      goto LABEL_19;
    goto LABEL_11;
  }
  if ( v7 == v10 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a3 + 8) + 20LL) & 7) == 1 )
    {
      sub_C3C790(a1, (_QWORD **)a3);
      v11 = (__int64)a1;
      goto LABEL_7;
    }
    goto LABEL_11;
  }
  if ( (*(_BYTE *)(a3 + 20) & 7) != 1 )
  {
LABEL_19:
    v12 = (__int64 *)a3;
    v13 = a1;
LABEL_20:
    sub_C33EB0(v13, v12);
    return a1;
  }
  sub_C33EB0(a1, (__int64 *)a3);
  v11 = (__int64)a1;
LABEL_7:
  if ( v7 == (_QWORD *)*a1 )
    v11 = a1[1];
  sub_C39170(v11);
  return a1;
}

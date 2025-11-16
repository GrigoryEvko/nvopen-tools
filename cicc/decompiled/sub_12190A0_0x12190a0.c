// Function: sub_12190A0
// Address: 0x12190a0
//
__int64 __fastcall sub_12190A0(__int64 a1, __int64 **a2, int *a3, char a4)
{
  unsigned int v6; // eax
  int v7; // eax
  int v8; // eax
  unsigned int v9; // r13d
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned int v13; // r15d
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  _BOOL8 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  char v22; // al
  char v23; // al
  size_t v24; // r15
  const void *v25; // r13
  int v26; // eax
  unsigned int v27; // r9d
  _QWORD *v28; // r10
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // r9d
  _QWORD *v32; // r10
  _QWORD *v33; // rcx
  __int64 *v34; // rax
  __int64 *v35; // rax
  const char *v36; // rax
  unsigned __int64 v37; // rsi
  const char *v38; // rax
  unsigned __int64 v39; // rsi
  _QWORD *v40; // [rsp+8h] [rbp-98h]
  _QWORD *v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  unsigned int v43; // [rsp+18h] [rbp-88h]
  __int64 v44; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+20h] [rbp-80h]
  unsigned __int64 v47; // [rsp+28h] [rbp-78h]
  int v48; // [rsp+3Ch] [rbp-64h] BYREF
  int v49[8]; // [rsp+40h] [rbp-60h] BYREF
  char v50; // [rsp+60h] [rbp-40h]
  char v51; // [rsp+61h] [rbp-3Fh]

  v47 = *(_QWORD *)(a1 + 232);
  v6 = *(_DWORD *)(a1 + 240);
  if ( v6 == 63 )
  {
    if ( !(unsigned __int8)sub_1219730() )
      goto LABEL_7;
    return 1;
  }
  if ( v6 <= 0x3F )
  {
    switch ( v6 )
    {
      case 8u:
        if ( !(unsigned __int8)sub_121A2D0(a1, a2, 0) )
          goto LABEL_7;
        return 1;
      case 0xAu:
        v7 = sub_1205200(a1 + 176);
        *(_DWORD *)(a1 + 240) = v7;
        if ( v7 == 8 )
        {
          if ( !(unsigned __int8)sub_121A2D0(a1, a2, 1)
            && !(unsigned __int8)sub_120AFE0(a1, 11, "expected '>' at end of packed struct") )
          {
            goto LABEL_7;
          }
        }
        else if ( !(unsigned __int8)sub_121A490(a1, a2, 1) )
        {
LABEL_7:
          v8 = *(_DWORD *)(a1 + 240);
          goto LABEL_8;
        }
        return 1;
      case 6u:
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( !(unsigned __int8)sub_121A490(a1, a2, 0) )
          goto LABEL_7;
        return 1;
    }
LABEL_51:
    v9 = 1;
    sub_11FD800(a1 + 176, v47, (__int64)a3, 1);
    return v9;
  }
  if ( v6 == 510 )
  {
    v24 = *(_QWORD *)(a1 + 256);
    v25 = *(const void **)(a1 + 248);
    v26 = sub_C92610();
    v27 = sub_C92740(a1 + 928, v25, v24, v26);
    v28 = (_QWORD *)(*(_QWORD *)(a1 + 928) + 8LL * v27);
    v29 = *v28;
    if ( *v28 )
    {
      if ( v29 != -8 )
      {
LABEL_58:
        v21 = *(_QWORD *)(v29 + 8);
        if ( !v21 )
        {
          v46 = v29;
          v21 = sub_BCC840(*(_QWORD **)a1, *(const void **)(a1 + 248), *(_QWORD *)(a1 + 256));
          *(_QWORD *)(v46 + 8) = v21;
          *(_QWORD *)(v46 + 16) = *(_QWORD *)(a1 + 232);
        }
        goto LABEL_60;
      }
      --*(_DWORD *)(a1 + 944);
    }
    v41 = v28;
    v43 = v27;
    v30 = sub_C7D670(v24 + 25, 8);
    v31 = v43;
    v32 = v41;
    v33 = (_QWORD *)v30;
    if ( v24 )
    {
      v40 = (_QWORD *)v30;
      memcpy((void *)(v30 + 24), v25, v24);
      v31 = v43;
      v32 = v41;
      v33 = v40;
    }
    *((_BYTE *)v33 + v24 + 24) = 0;
    *v33 = v24;
    v33[1] = 0;
    v33[2] = 0;
    *v32 = v33;
    ++*(_DWORD *)(a1 + 940);
    v34 = (__int64 *)(*(_QWORD *)(a1 + 928) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 928), v31));
    v29 = *v34;
    if ( !*v34 || v29 == -8 )
    {
      v35 = v34 + 1;
      do
      {
        do
          v29 = *v35++;
        while ( !v29 );
      }
      while ( v29 == -8 );
    }
    goto LABEL_58;
  }
  if ( v6 != 527 )
  {
    if ( v6 != 504 )
      goto LABEL_51;
    v12 = *(_QWORD *)(a1 + 968);
    v13 = *(_DWORD *)(a1 + 280);
    v14 = a1 + 960;
    if ( !v12 )
      goto LABEL_23;
    do
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v12 + 16);
        v20 = *(_QWORD *)(v12 + 24);
        if ( v13 <= *(_DWORD *)(v12 + 32) )
          break;
        v12 = *(_QWORD *)(v12 + 24);
        if ( !v20 )
          goto LABEL_31;
      }
      v14 = v12;
      v12 = *(_QWORD *)(v12 + 16);
    }
    while ( v19 );
LABEL_31:
    if ( a1 + 960 == v14 || v13 < *(_DWORD *)(v14 + 32) )
    {
LABEL_23:
      v44 = v14;
      v42 = a1 + 960;
      v15 = sub_22077B0(56);
      *(_DWORD *)(v15 + 32) = v13;
      v14 = v15;
      *(_QWORD *)(v15 + 40) = 0;
      *(_QWORD *)(v15 + 48) = 0;
      v16 = sub_1216560((_QWORD *)(a1 + 952), v44, (unsigned int *)(v15 + 32));
      if ( v17 )
      {
        v18 = v16 || v42 == v17 || v13 < *(_DWORD *)(v17 + 32);
        sub_220F040(v18, v14, v17, v42);
        ++*(_QWORD *)(a1 + 992);
      }
      else
      {
        v45 = v16;
        j_j___libc_free_0(v14, 56);
        v14 = v45;
      }
    }
    v21 = *(_QWORD *)(v14 + 40);
    if ( !v21 )
    {
      v21 = sub_BCC900(*(_QWORD **)a1);
      *(_QWORD *)(v14 + 40) = v21;
      *(_QWORD *)(v14 + 48) = *(_QWORD *)(a1 + 232);
    }
LABEL_60:
    *a2 = (__int64 *)v21;
    v8 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v8;
    goto LABEL_8;
  }
  v11 = a1 + 176;
  *a2 = *(__int64 **)(a1 + 288);
  v8 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v8;
  if ( *((_BYTE *)*a2 + 8) != 14 )
    goto LABEL_8;
  v9 = sub_1212650(a1, &v48, 0);
  if ( (_BYTE)v9 )
    return v9;
  *a2 = (__int64 *)sub_BCE3C0(*(__int64 **)a1, v48);
  v8 = *(_DWORD *)(a1 + 240);
  if ( v8 == 5 )
  {
    v51 = 1;
    v38 = "ptr* is invalid - use ptr instead";
LABEL_77:
    *(_QWORD *)v49 = v38;
    v39 = *(_QWORD *)(a1 + 232);
    v50 = 3;
    sub_11FD800(v11, v39, (__int64)v49, 1);
    return 1;
  }
  if ( v8 != 12 )
    return v9;
LABEL_8:
  while ( v8 == 12 )
  {
LABEL_42:
    if ( (unsigned __int8)sub_1219020(a1, a2) )
      return 1;
    v8 = *(_DWORD *)(a1 + 240);
  }
  while ( 1 )
  {
    if ( v8 == 94 )
    {
      v23 = *((_BYTE *)*a2 + 8);
      if ( v23 == 8 )
      {
        v51 = 1;
        v36 = "basic block pointers are invalid";
      }
      else if ( v23 == 7 )
      {
        v51 = 1;
        v36 = "pointers to void are invalid; use i8* instead";
      }
      else
      {
        if ( sub_BCBD20((__int64)*a2) )
        {
          if ( !(unsigned __int8)sub_1212650(a1, v49, 0)
            && !(unsigned __int8)sub_120AFE0(a1, 5, "expected '*' in address space") )
          {
            *a2 = (__int64 *)sub_BCE3C0(*(__int64 **)a1, v49[0]);
            v8 = *(_DWORD *)(a1 + 240);
            goto LABEL_8;
          }
          return 1;
        }
        v51 = 1;
        v36 = "pointer to this type is invalid";
      }
      v37 = *(_QWORD *)(a1 + 232);
      *(_QWORD *)v49 = v36;
      v50 = 3;
      sub_11FD800(a1 + 176, v37, (__int64)v49, 1);
      return 1;
    }
    if ( v8 != 5 )
      break;
    v11 = a1 + 176;
    v22 = *((_BYTE *)*a2 + 8);
    if ( v22 == 8 )
    {
      v51 = 1;
      v38 = "basic block pointers are invalid";
      goto LABEL_77;
    }
    if ( v22 == 7 )
    {
      v51 = 1;
      v38 = "pointers to void are invalid - use i8* instead";
      goto LABEL_77;
    }
    if ( !sub_BCBD20((__int64)*a2) )
    {
      v51 = 1;
      v38 = "pointer to this type is invalid";
      goto LABEL_77;
    }
    *a2 = (__int64 *)sub_BCE3C0(*(__int64 **)a1, 0);
    v8 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v8;
    if ( v8 == 12 )
      goto LABEL_42;
  }
  v9 = 0;
  if ( !a4 && *((_BYTE *)*a2 + 8) == 7 )
  {
    v51 = 1;
    a3 = v49;
    *(_QWORD *)v49 = "void type only allowed for function results";
    v50 = 3;
    goto LABEL_51;
  }
  return v9;
}

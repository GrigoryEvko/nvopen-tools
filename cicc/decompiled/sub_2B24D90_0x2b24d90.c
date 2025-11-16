// Function: sub_2B24D90
// Address: 0x2b24d90
//
_BOOL8 __fastcall sub_2B24D90(char *a1, char *a2, __int64 a3, __int64 a4)
{
  int v4; // r13d
  __int64 v7; // rax
  __int64 v8; // rax
  char *v9; // r14
  char *v10; // r15
  __int64 v11; // rcx
  char *v12; // rax
  bool v13; // al
  __int64 v14; // rax
  bool v15; // al
  unsigned int v16; // esi
  __int64 (__fastcall *v17)(__int64, char *); // r8
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 (__fastcall *v20)(__int64, char *); // r8
  __int64 v21; // rax
  char *v22; // rax
  char *v23; // rax
  char v24; // [rsp-79h] [rbp-79h]
  __int64 v25; // [rsp-78h] [rbp-78h]
  __int64 v27; // [rsp-70h] [rbp-70h]
  bool v28; // [rsp-68h] [rbp-68h]
  bool v29; // [rsp-68h] [rbp-68h]
  unsigned __int64 *v30; // [rsp-58h] [rbp-58h] BYREF
  __int64 v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-48h] [rbp-48h]
  __int64 v33; // [rsp-40h] [rbp-40h]

  if ( *((_QWORD *)a1 + 5) != *((_QWORD *)a2 + 5) )
    return 0;
  if ( *((_QWORD *)a1 + 1) != *((_QWORD *)a2 + 1) )
    return 0;
  v7 = *((_QWORD *)a1 + 2);
  if ( !v7 || *(_QWORD *)(v7 + 8) )
  {
    v8 = *((_QWORD *)a2 + 2);
    if ( !v8 || *(_QWORD *)(v8 + 8) )
      return 0;
  }
  v31 = sub_2B18C70(a1, 0);
  v32 = sub_2B18C70(a2, 0);
  if ( !BYTE4(v31) )
    return 0;
  v24 = BYTE4(v32);
  if ( !BYTE4(v32) )
    return 0;
  v9 = a2;
  v10 = a1;
  sub_B48880((__int64 *)&v30, *(_DWORD *)(*((_QWORD *)a1 + 1) + 32LL), 0);
  v11 = a4;
  do
  {
    while ( 1 )
    {
      LOBYTE(v4) = a2 != v10 && v10 != 0;
      if ( (_BYTE)v4 )
      {
        v25 = v11;
        v29 = a1 != v9 && v9 != 0;
        v33 = sub_2B18C70(v10, 0);
        if ( BYTE4(v33) )
          v19 = v33;
        else
          v19 = v32;
        v4 = sub_2B0D930((unsigned __int64)v30, v19);
        sub_2B0D980((__int64 *)&v30, v19);
        v11 = v25;
        if ( a1 != v10 )
        {
          v21 = *((_QWORD *)v10 + 2);
          if ( !v21 || *(_QWORD *)(v21 + 8) )
          {
            v10 = 0;
            if ( !v29 )
              goto LABEL_37;
            goto LABEL_32;
          }
        }
        if ( (_BYTE)v4 )
        {
          if ( !v29 )
            goto LABEL_21;
          v4 = v29;
LABEL_31:
          v10 = 0;
          goto LABEL_32;
        }
        v22 = (char *)v20(v25, v10);
        v11 = v25;
        v10 = v22;
        if ( v22 )
        {
          if ( *v22 != 91 )
          {
            if ( !v29 )
            {
              v12 = v9;
              v10 = 0;
LABEL_14:
              if ( !v12 )
                goto LABEL_21;
              if ( a1 == v9 && !v10 )
              {
LABEL_27:
                v14 = *((_QWORD *)a1 + 2);
                if ( v14 )
                  goto LABEL_28;
LABEL_21:
                v15 = 0;
                goto LABEL_22;
              }
LABEL_17:
              v13 = v9 == 0;
              goto LABEL_18;
            }
            goto LABEL_31;
          }
          if ( !v29 )
            goto LABEL_17;
        }
        else
        {
          v12 = v9;
          if ( !v29 )
            goto LABEL_14;
        }
      }
      else if ( a1 == v9 || v9 == 0 )
      {
        v12 = (char *)((unsigned __int64)v9 | (unsigned __int64)v10);
        goto LABEL_14;
      }
LABEL_32:
      v27 = v11;
      v33 = sub_2B18C70(v9, 0);
      if ( BYTE4(v33) )
        v16 = v33;
      else
        v16 = v31;
      v4 |= sub_2B0D930((unsigned __int64)v30, v16);
      sub_2B0D980((__int64 *)&v30, v16);
      v11 = v27;
      if ( a2 != v9 )
      {
        v18 = *((_QWORD *)v9 + 2);
        if ( !v18 || *(_QWORD *)(v18 + 8) )
        {
          v9 = 0;
LABEL_37:
          if ( (_BYTE)v4 )
            goto LABEL_21;
          v12 = (char *)((unsigned __int64)v10 | (unsigned __int64)v9);
          goto LABEL_14;
        }
      }
      if ( (_BYTE)v4 )
        goto LABEL_21;
      v23 = (char *)v17(v27, v9);
      v11 = v27;
      v9 = v23;
      if ( !v23 || *v23 != 91 )
        break;
      if ( !v10 && a1 == v23 )
        goto LABEL_27;
    }
    if ( !v10 )
      goto LABEL_21;
    v13 = v24;
    v9 = 0;
LABEL_18:
    ;
  }
  while ( a2 != v10 || !v13 );
  v14 = *((_QWORD *)a2 + 2);
  if ( !v14 )
    goto LABEL_21;
LABEL_28:
  v15 = *(_QWORD *)(v14 + 8) == 0;
LABEL_22:
  v28 = v15;
  sub_228BF40(&v30);
  return v28;
}

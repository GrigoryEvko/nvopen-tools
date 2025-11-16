// Function: sub_14A4980
// Address: 0x14a4980
//
__int64 __fastcall sub_14A4980(__int64 a1, int a2, unsigned int a3)
{
  __int64 result; // rax
  bool v6; // bl
  __int64 *v7; // r14
  __int64 *v8; // r15
  __int64 v9; // rax
  bool v10; // dl
  __int64 *v11; // rcx
  __int64 v12; // r8
  bool v13; // cl
  unsigned int v14; // r9d
  char v15; // dl
  __int64 *v16; // rdi
  __int64 *v17; // rdi
  bool v18; // si
  char v19; // al
  char v20; // al
  bool v21; // [rsp-AAh] [rbp-AAh]
  bool v22; // [rsp-A9h] [rbp-A9h]
  bool v23; // [rsp-A9h] [rbp-A9h]
  __int64 v24; // [rsp-A8h] [rbp-A8h]
  __int64 v25; // [rsp-A8h] [rbp-A8h]
  __int64 v26; // [rsp-A8h] [rbp-A8h]
  __int64 v27; // [rsp-A0h] [rbp-A0h]
  unsigned int v28; // [rsp-A0h] [rbp-A0h]
  unsigned int v29; // [rsp-A0h] [rbp-A0h]
  unsigned int v30; // [rsp-A0h] [rbp-A0h]
  int v31; // [rsp-98h] [rbp-98h] BYREF
  __int64 v32; // [rsp-90h] [rbp-90h]
  __int64 v33; // [rsp-88h] [rbp-88h]
  unsigned int v34; // [rsp-80h] [rbp-80h]
  bool v35; // [rsp-78h] [rbp-78h]
  _DWORD v36[8]; // [rsp-68h] [rbp-68h] BYREF
  char v37; // [rsp-48h] [rbp-48h]

  result = 0;
  if ( a1 )
  {
    sub_14A4410((__int64)&v31, a1);
    v6 = v35;
    if ( !v35 )
      return 0;
    v7 = (__int64 *)v32;
    if ( *(_BYTE *)(v32 + 16) == 85 )
    {
      v8 = (__int64 *)v33;
      if ( *(_BYTE *)(v33 + 16) != 85 )
      {
        if ( a2 )
          return 0;
        v9 = *(_QWORD *)(v32 - 72);
        v13 = v35;
        v12 = v33;
        v17 = (__int64 *)v32;
        v10 = v9 != 0;
        v8 = 0;
        v6 = 0;
        goto LABEL_33;
      }
      v12 = *(_QWORD *)(v33 - 72);
      v9 = *(_QWORD *)(v32 - 72);
      v10 = a2 == 0;
      if ( v12 )
      {
        if ( !v9 )
        {
          v11 = (__int64 *)v32;
          v9 = *(_QWORD *)(v33 - 72);
          v6 = 0;
LABEL_10:
          if ( !v10 || v32 != v9 )
            return 0;
          v7 = v11;
          v12 = v9;
          v13 = 0;
          goto LABEL_13;
        }
        v6 = 0;
        v13 = 0;
        if ( v12 != v9 )
          return 0;
LABEL_13:
        v14 = a2 + 1;
        if ( a2 + 1 != a3 )
        {
          v21 = v13;
          v22 = v10;
          v27 = v12;
          sub_14A4410((__int64)v36, v12);
          if ( !v37 )
            return 0;
          v12 = v27;
          v14 = a2 + 1;
          v10 = v22;
          v13 = v21;
          if ( v34 != v36[6] || v31 != v36[0] )
            return 0;
        }
        if ( v6 && v10 )
          goto LABEL_19;
        if ( v7 )
        {
          v23 = v13;
          v25 = v12;
          v29 = v14;
          v19 = sub_14A0CC0(v7, 1u, a2);
          v14 = v29;
          v12 = v25;
          if ( v19 )
          {
LABEL_19:
            if ( !v8 )
              return 0;
            v24 = v12;
            v15 = a2;
            v16 = v8;
            v28 = v14;
            goto LABEL_21;
          }
          if ( v23 )
          {
LABEL_40:
            v24 = v12;
            v15 = a2;
            v16 = v7;
            v28 = v14;
LABEL_21:
            if ( (unsigned __int8)sub_14A0CC0(v16, 0, v15) )
            {
              if ( v28 == a3 )
                return v34;
              else
                return sub_14A4980(v24, v28, a3);
            }
            return 0;
          }
        }
        else if ( v13 )
        {
          return 0;
        }
        v26 = v12;
        v30 = v14;
        if ( !v8 )
          return 0;
        v20 = sub_14A0CC0(v8, 1u, a2);
        v14 = v30;
        v12 = v26;
        if ( !v20 || !v7 )
          return 0;
        goto LABEL_40;
      }
      v17 = (__int64 *)v32;
      v18 = a2 == 0;
      v12 = v33;
      v6 = 0;
    }
    else
    {
      if ( a2 )
        return 0;
      v8 = (__int64 *)v33;
      if ( *(_BYTE *)(v33 + 16) != 85 )
        return 0;
      v9 = *(_QWORD *)(v33 - 72);
      if ( v9 )
      {
        v10 = v35;
        v11 = 0;
        goto LABEL_10;
      }
      v17 = 0;
      v18 = v35;
      v12 = v33;
    }
    v10 = v9 != 0;
    v13 = 0;
    if ( !v18 )
      return 0;
LABEL_33:
    if ( !v10 )
      return 0;
    if ( v9 )
    {
      if ( v12 != v9 )
        return 0;
    }
    else
    {
      v12 = v32;
    }
    v7 = v17;
    goto LABEL_13;
  }
  return result;
}

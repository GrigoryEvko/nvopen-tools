// Function: sub_2EE6520
// Address: 0x2ee6520
//
__int64 __fastcall sub_2EE6520(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r14
  unsigned int v5; // r14d
  int *v7; // rax
  unsigned int v8; // r15d
  int v9; // eax
  __int64 v10; // r14
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  _DWORD *v17; // rax
  int v18; // r15d
  __int64 v19; // r14
  __int64 *v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  int v23; // [rsp+Ch] [rbp-94h]
  unsigned __int64 v24; // [rsp+10h] [rbp-90h] BYREF
  char v25; // [rsp+20h] [rbp-80h]
  unsigned __int64 v26[2]; // [rsp+30h] [rbp-70h] BYREF
  char v27; // [rsp+40h] [rbp-60h]
  __int64 *v28; // [rsp+50h] [rbp-50h] BYREF
  __int64 v29; // [rsp+58h] [rbp-48h]
  char v30; // [rsp+60h] [rbp-40h]

  v4 = *a1;
  if ( (unsigned __int8)sub_B2D610(*a1, 47) )
    return 1;
  v5 = sub_B2D610(v4, 18);
  if ( (_BYTE)v5 )
    return 1;
  if ( a2 )
  {
    if ( a3 )
    {
      v7 = *(int **)(a2 + 8);
      if ( v7 )
      {
        if ( LOBYTE(qword_4F91668[8]) )
          return 1;
        v8 = LOBYTE(qword_4F91BA8[8]);
        if ( !LOBYTE(qword_4F91BA8[8]) )
          return v5;
        if ( !LOBYTE(qword_4F919E8[8]) )
        {
          v9 = *v7;
          if ( v9 )
          {
            if ( v9 == 2 )
            {
              if ( !(unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91828[8])
                || (unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91748[8]) )
              {
LABEL_26:
                if ( !*(_QWORD *)(a2 + 8) )
                  return v5;
                goto LABEL_27;
              }
              if ( !LOBYTE(qword_4F91AC8[8]) )
                goto LABEL_38;
              goto LABEL_47;
            }
LABEL_13:
            if ( !LOBYTE(qword_4F91AC8[8]) )
              goto LABEL_14;
LABEL_47:
            if ( !(unsigned __int8)sub_D84430(a2) )
              goto LABEL_26;
LABEL_38:
            v17 = *(_DWORD **)(a2 + 8);
            if ( !v17 )
              return v8;
            if ( *v17 != 2 )
            {
LABEL_14:
              v23 = qword_4F91588[8];
              sub_2E7A420((__int64)&v28, a2, a1);
              if ( v30 && sub_D85370(a2, v23, (unsigned __int64)v28) )
                return 0;
              v10 = a1[41];
              if ( (__int64 *)v10 != a1 + 40 )
              {
                while ( 1 )
                {
                  v11 = sub_2E39F50(a3, v10);
                  v26[1] = v12;
                  v26[0] = (unsigned __int64)v11;
                  if ( (_BYTE)v12 )
                  {
                    if ( sub_D85370(a2, v23, v26[0]) )
                      break;
                  }
                  v10 = *(_QWORD *)(v10 + 8);
                  if ( a1 + 40 == (__int64 *)v10 )
                    return v8;
                }
                return 0;
              }
              return v8;
            }
            v18 = qword_4F914A8[8];
            sub_2E7A420((__int64)v26, a2, a1);
            if ( !v27 || (LOBYTE(v22) = sub_D853A0(a2, v18, v26[0]), v5 = v22, (_BYTE)v22) )
            {
              v19 = a1[41];
              if ( (__int64 *)v19 != a1 + 40 )
              {
                while ( 1 )
                {
                  v20 = sub_2E39F50(a3, v19);
                  v29 = v21;
                  v28 = v20;
                  if ( !(_BYTE)v21 || !sub_D853A0(a2, v18, (unsigned __int64)v28) )
                    break;
                  v19 = *(_QWORD *)(v19 + 8);
                  if ( a1 + 40 == (__int64 *)v19 )
                    return 1;
                }
                return 0;
              }
              return 1;
            }
            return v5;
          }
          if ( !LOBYTE(qword_4F91908[8]) )
            goto LABEL_13;
        }
LABEL_27:
        sub_2E7A420((__int64)&v24, a2, a1);
        if ( v25 )
        {
          LOBYTE(v16) = sub_D84450(a2, v24);
          v5 = v16;
          if ( !(_BYTE)v16 )
            return v5;
        }
        v13 = a1[41];
        if ( (__int64 *)v13 != a1 + 40 )
        {
          while ( 1 )
          {
            v14 = sub_2E39F50(a3, v13);
            v29 = v15;
            v28 = v14;
            if ( !(_BYTE)v15 || !sub_D84450(a2, (unsigned __int64)v28) )
              break;
            v13 = *(_QWORD *)(v13 + 8);
            if ( a1 + 40 == (__int64 *)v13 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
    }
  }
  return v5;
}

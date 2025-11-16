// Function: sub_5CEC90
// Address: 0x5cec90
//
void __fastcall sub_5CEC90(_QWORD *a1, __int64 a2, char a3)
{
  __int64 v3; // r15
  _QWORD *v5; // rbx
  _QWORD **v6; // rax
  _BYTE *v7; // r14
  int v8; // eax
  int *v9; // r8
  char *v10; // rax
  int *v11; // [rsp+8h] [rbp-78h]
  int v12; // [rsp+18h] [rbp-68h] BYREF
  int v13; // [rsp+1Ch] [rbp-64h] BYREF
  int v14; // [rsp+20h] [rbp-60h] BYREF
  int v15; // [rsp+24h] [rbp-5Ch] BYREF
  int v16; // [rsp+28h] [rbp-58h] BYREF
  int v17; // [rsp+2Ch] [rbp-54h] BYREF
  int v18; // [rsp+30h] [rbp-50h] BYREF
  int v19; // [rsp+34h] [rbp-4Ch] BYREF
  int v20; // [rsp+38h] [rbp-48h] BYREF
  int v21; // [rsp+3Ch] [rbp-44h] BYREF
  int v22; // [rsp+40h] [rbp-40h] BYREF
  int v23; // [rsp+44h] [rbp-3Ch] BYREF
  int v24; // [rsp+48h] [rbp-38h] BYREF
  int v25; // [rsp+4Ch] [rbp-34h] BYREF

  v3 = a2;
  v5 = a1;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  if ( a2 )
  {
    v6 = (_QWORD **)sub_5CEB70(a2, a3);
    if ( !v6 )
    {
      MEMORY[0] = a1;
      BUG();
    }
    if ( *v6 )
      v6 = sub_5CB9F0(v6);
    *v6 = a1;
  }
  if ( a1 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v5;
        v5 = (_QWORD *)*v5;
        if ( (v7[11] & 2) == 0 || (unsigned __int8)(v7[10] - 2) <= 1u )
          break;
        if ( !v5 )
          return;
      }
      switch ( v7[8] )
      {
        case 'V':
          v8 = v19;
          v9 = &v19;
          goto LABEL_13;
        case 'W':
          v8 = v13;
          v9 = &v13;
          goto LABEL_13;
        case 'X':
          v8 = v12;
          v9 = &v12;
          goto LABEL_13;
        case 'Y':
          v8 = v15;
          v9 = &v15;
          goto LABEL_13;
        case 'Z':
          v8 = v18;
          v9 = &v18;
          goto LABEL_13;
        case '[':
          v8 = v14;
          v9 = &v14;
          goto LABEL_13;
        case '\\':
          v8 = v20;
          v9 = &v20;
          goto LABEL_13;
        case ']':
          v8 = v21;
          v9 = &v21;
          goto LABEL_13;
        case '^':
          v8 = v22;
          v9 = &v22;
          goto LABEL_13;
        case '_':
          v8 = v16;
          v9 = &v16;
          goto LABEL_13;
        case 'f':
          v8 = v17;
          v9 = &v17;
          goto LABEL_13;
        case 'k':
          v8 = v23;
          v9 = &v23;
          goto LABEL_13;
        case 'l':
          v9 = &v24;
          if ( v24 )
            goto LABEL_18;
          goto LABEL_14;
        case 'r':
          v8 = v25;
          v9 = &v25;
LABEL_13:
          if ( v8 )
          {
LABEL_18:
            v11 = v9;
            v10 = sub_5C79F0((__int64)v7);
            sub_6849F0(4, 3646, v7 + 56, v10);
            v9 = v11;
          }
LABEL_14:
          *v9 = 1;
          break;
        default:
          break;
      }
      v3 = sub_5CD370((__int64)v7, v3, a3);
    }
    while ( v5 );
  }
}

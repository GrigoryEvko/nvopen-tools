// Function: sub_E17E90
// Address: 0xe17e90
//
__int64 __fastcall sub_E17E90(__int64 **a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r14
  char v5; // r9
  __int64 v6; // r15
  __int64 v7; // rax
  char *v8; // rdx
  char *v9; // rsi
  char v10; // al
  int v11; // r8d
  int v13; // r8d
  int v14; // [rsp+Ch] [rbp-54h]
  char v15; // [rsp+Ch] [rbp-54h]
  char v16; // [rsp+Ch] [rbp-54h]
  int v17; // [rsp+Ch] [rbp-54h]
  char v18[80]; // [rsp+10h] [rbp-50h] BYREF

  v3 = a2[1];
  sub_E14360((__int64)a2, 34);
  v4 = *a1;
  v5 = 0;
  v6 = (__int64)&(*a1)[(_QWORD)a1[1]];
  if ( *a1 != (__int64 *)v6 )
  {
    while ( 1 )
    {
      v7 = *v4;
      if ( *(_BYTE *)(*v4 + 8) != 77 )
      {
LABEL_18:
        a2[1] = v3;
        return 0;
      }
      v8 = *(char **)(v7 + 40);
      v9 = &v8[*(_QWORD *)(v7 + 32)];
      if ( v8 == v9 )
        break;
      v10 = *v8;
      if ( (unsigned __int8)(*v8 - 48) > 9u )
        goto LABEL_18;
      v11 = 0;
      while ( 1 )
      {
        ++v8;
        v11 = (char)(v10 - 48) + 10 * v11;
        if ( v8 == v9 )
          break;
        v10 = *v8;
        if ( (unsigned __int8)(*v8 - 48) > 9u || v11 > 25 )
          goto LABEL_18;
      }
      if ( v11 > 255 )
        goto LABEL_18;
      if ( v5 && ((unsigned int)(v11 - 48) <= 9 || (unsigned int)(v11 - 97) <= 5 || (unsigned int)(v11 - 65) <= 5) )
      {
        v14 = v11;
        sub_E12F20(a2, 2u, "\"\"");
        v11 = v14;
      }
      if ( v11 <= 34 )
      {
        if ( v11 > 6 )
        {
          switch ( v11 )
          {
            case 7:
              sub_E12F20(a2, 2u, "\\a");
              v5 = 0;
              break;
            case 8:
              sub_E12F20(a2, 2u, "\\b");
              v5 = 0;
              break;
            case 9:
              sub_E12F20(a2, 2u, "\\t");
              v5 = 0;
              break;
            case 10:
              sub_E12F20(a2, 2u, "\\n");
              v5 = 0;
              break;
            case 11:
              sub_E12F20(a2, 2u, "\\v");
              v5 = 0;
              break;
            case 12:
              sub_E12F20(a2, 2u, "\\f");
              v5 = 0;
              break;
            case 13:
              sub_E12F20(a2, 2u, "\\r");
              v5 = 0;
              break;
            case 34:
              sub_E12F20(a2, 2u, "\\\"");
              v5 = 0;
              break;
            default:
              if ( v11 > 31 )
                goto LABEL_29;
              v17 = v11;
              strcpy(v18, "0123456789ABCDEF");
              sub_E14360((__int64)a2, 92);
              sub_E14360((__int64)a2, 120);
              v13 = v17;
              if ( v17 > 15 )
                goto LABEL_27;
              goto LABEL_31;
          }
          goto LABEL_21;
        }
LABEL_25:
        v15 = v11;
        strcpy(v18, "0123456789ABCDEF");
        sub_E14360((__int64)a2, 92);
        LOBYTE(v13) = v15;
        goto LABEL_31;
      }
      if ( v11 == 92 )
      {
        sub_E12F20(a2, 2u, "\\\\");
        v5 = 0;
      }
      else
      {
        if ( v11 == 127 )
        {
          strcpy(v18, "0123456789ABCDEF");
          sub_E14360((__int64)a2, 92);
          sub_E14360((__int64)a2, 120);
          v13 = 127;
LABEL_27:
          v16 = v13;
          sub_E14360((__int64)a2, v18[v13 >> 4]);
          LOBYTE(v13) = v16;
          goto LABEL_31;
        }
LABEL_29:
        sub_E14360((__int64)a2, v11);
        v5 = 0;
      }
LABEL_21:
      if ( (__int64 *)v6 == ++v4 )
        goto LABEL_22;
    }
    if ( v5 )
    {
      LOBYTE(v11) = 0;
      goto LABEL_25;
    }
    strcpy(v18, "0123456789ABCDEF");
    sub_E14360((__int64)a2, 92);
    LOBYTE(v13) = 0;
LABEL_31:
    sub_E14360((__int64)a2, v18[v13 & 0xF]);
    v5 = 1;
    goto LABEL_21;
  }
LABEL_22:
  sub_E14360((__int64)a2, 34);
  return 1;
}

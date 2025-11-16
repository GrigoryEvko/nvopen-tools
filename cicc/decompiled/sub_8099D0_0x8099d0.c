// Function: sub_8099D0
// Address: 0x8099d0
//
const char *__fastcall sub_8099D0(__int64 a1, unsigned __int8 a2)
{
  char i; // al
  unsigned __int8 v3; // al
  char *v4; // rax
  char v6; // r12
  const char *v7; // [rsp+0h] [rbp-30h]
  char *v8; // [rsp+8h] [rbp-28h]
  char *v9; // [rsp+10h] [rbp-20h]
  char *v10; // [rsp+18h] [rbp-18h]

  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(a1 + 140) )
    a1 = *(_QWORD *)(a1 + 160);
  switch ( i )
  {
    case 3:
      v3 = *(_BYTE *)(a1 + 160);
      if ( v3 == 4 )
      {
        v7 = "__SVFloat64_t";
        v8 = "svfloat64x2_t";
        v9 = "svfloat64x3_t";
        v4 = "svfloat64x4_t";
        break;
      }
      if ( v3 > 4u )
      {
        if ( v3 == 9 )
        {
          v7 = "__SVBfloat16_t";
          v8 = "svbfloat16x2_t";
          v9 = "svbfloat16x3_t";
          v4 = "svbfloat16x4_t";
          break;
        }
      }
      else
      {
        if ( v3 == 1 )
        {
          v7 = "__SVFloat16_t";
          v8 = "svfloat16x2_t";
          v9 = "svfloat16x3_t";
          v4 = "svfloat16x4_t";
          break;
        }
        if ( v3 == 2 )
        {
          v7 = "__SVFloat32_t";
          v8 = "svfloat32x2_t";
          v9 = "svfloat32x3_t";
          v4 = "svfloat32x4_t";
          break;
        }
      }
LABEL_6:
      sub_721090();
    case 18:
      v7 = "__SVMFloat8_t";
      v8 = "svmfloat8x2_t";
      v9 = "svmfloat8x3_t";
      v4 = "svmfloat8x4_t";
      break;
    case 2:
      v6 = *(_BYTE *)(a1 + 160);
      if ( !(unsigned int)sub_8D29A0(a1) )
      {
        switch ( v6 )
        {
          case 1:
            v7 = "__SVInt8_t";
            v8 = "svint8x2_t";
            v9 = "svint8x3_t";
            v4 = "svint8x4_t";
            goto LABEL_12;
          case 2:
            v7 = "__SVUint8_t";
            v8 = "svuint8x2_t";
            v9 = "svuint8x3_t";
            v4 = "svuint8x4_t";
            goto LABEL_12;
          case 3:
            v7 = "__SVInt16_t";
            v8 = "svint16x2_t";
            v9 = "svint16x3_t";
            v4 = "svint16x4_t";
            goto LABEL_12;
          case 4:
            v7 = "__SVUint16_t";
            v8 = "svuint16x2_t";
            v9 = "svuint16x3_t";
            v4 = "svuint16x4_t";
            goto LABEL_12;
          case 5:
            v7 = "__SVInt32_t";
            v8 = "svint32x2_t";
            v9 = "svint32x3_t";
            v4 = "svint32x4_t";
            goto LABEL_12;
          case 6:
            v7 = "__SVUint32_t";
            v8 = "svuint32x2_t";
            v9 = "svuint32x3_t";
            v4 = "svuint32x4_t";
            goto LABEL_12;
          case 7:
            v7 = "__SVInt64_t";
            v8 = "svint64x2_t";
            v9 = "svint64x3_t";
            v4 = "svint64x4_t";
            goto LABEL_12;
          case 8:
            v7 = "__SVUint64_t";
            v8 = "svuint64x2_t";
            v9 = "svuint64x3_t";
            v4 = "svuint64x4_t";
            goto LABEL_12;
          default:
            goto LABEL_6;
        }
      }
      v9 = 0;
      v7 = "__SVBool_t";
      v8 = (char *)"svboolx2_t";
      v4 = (char *)"svboolx4_t";
      break;
    default:
      goto LABEL_6;
  }
LABEL_12:
  v10 = v4;
  return (&v7)[a2 - 1];
}

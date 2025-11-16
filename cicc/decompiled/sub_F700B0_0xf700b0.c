// Function: sub_F700B0
// Address: 0xf700b0
//
unsigned __int8 *__fastcall sub_F700B0(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 v8; // r13
  char v9; // bl
  unsigned int v10; // eax
  unsigned __int8 *result; // rax
  unsigned int v12; // eax
  unsigned int v13; // r14d
  __int64 v14; // r15
  void *v15; // rax
  void *v16; // rbx
  __int64 v17; // r13
  __int64 v18[11]; // [rsp-58h] [rbp-58h] BYREF

  v18[10] = v6;
  v18[7] = v8;
  v18[6] = v7;
  v18[5] = v5;
  v9 = a3;
  switch ( a1 )
  {
    case 387:
    case 388:
    case 389:
    case 394:
    case 395:
    case 396:
    case 401:
      v10 = sub_F6EED0(a1);
      return sub_AD93D0(v10, a2, 0, (v9 & 8) != 0);
    case 390:
    case 391:
      v13 = 1;
      goto LABEL_6;
    case 392:
    case 393:
      v13 = 0;
LABEL_6:
      v14 = sub_BCAC60(a2, a2, a3, a4, a5);
      if ( ((a1 - 391) & 0xFFFFFFFD) != 0 && (v9 & 2) == 0 )
      {
        result = sub_AD9140(a2, v13, 0);
      }
      else if ( (v9 & 4) != 0 )
      {
        v15 = sub_C33340();
        v16 = v15;
        if ( (void *)v14 == v15 )
          sub_C3C500(v18, (__int64)v15);
        else
          sub_C373C0(v18, v14);
        if ( (void *)v18[0] == v16 )
          sub_C3CF90((__int64)v18, v13);
        else
          sub_C35910((__int64)v18, v13);
        v17 = sub_AD8F10(a2, v18);
        sub_91D830(v18);
        result = (unsigned __int8 *)v17;
      }
      else
      {
        result = (unsigned __int8 *)sub_AD9500(a2, v13);
      }
      break;
    case 397:
    case 398:
    case 399:
    case 400:
      v12 = sub_F6EFA0(a1);
      result = (unsigned __int8 *)sub_AD66B0(v12, a2);
      break;
    default:
      BUG();
  }
  return result;
}

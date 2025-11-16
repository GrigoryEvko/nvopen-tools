// Function: sub_818B40
// Address: 0x818b40
//
void __fastcall sub_818B40(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  char v8; // bl
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  const __m128i *v14; // rdi
  __int64 v15; // rsi
  const __m128i *v16; // [rsp+8h] [rbp-28h] BYREF

  v8 = a3;
  if ( (a3 & 0xFD) != 5 || dword_4D0425C && qword_4F077A8 <= 0x9C3Fu )
  {
    switch ( (char)a3 )
    {
      case 5:
        if ( a2 )
          goto LABEL_20;
        goto LABEL_26;
      case 7:
        goto LABEL_8;
      case 8:
        v11 = *a4;
        v12 = qword_4F18BE0;
        if ( a2 )
        {
          *a4 = v11 + 12;
          sub_8238B0(v12, "v19__uuidofe", 12);
          goto LABEL_14;
        }
        *a4 = v11 + 11;
        sub_8238B0(v12, "v18__uuidof", 11);
        goto LABEL_16;
      case 9:
        v13 = qword_4F18BE0;
        *a4 += 2;
        if ( a2 )
        {
          sub_8238B0(v13, "te", 2);
          goto LABEL_14;
        }
        sub_8238B0(v13, "ti", 2);
        break;
      case 10:
        *a4 += 2;
        sub_8238B0(qword_4F18BE0, "nx", 2);
        if ( !a2 )
          goto LABEL_16;
        goto LABEL_14;
      default:
        goto LABEL_29;
    }
    goto LABEL_16;
  }
  if ( a2 )
  {
    if ( (unsigned int)sub_731EE0(a2, a2, a3, dword_4D0425C, a5, a6) )
    {
      switch ( v8 )
      {
        case 5:
LABEL_20:
          *a4 += 2;
          sub_8238B0(qword_4F18BE0, "sz", 2);
          goto LABEL_14;
        case 7:
          goto LABEL_8;
        default:
          goto LABEL_29;
      }
    }
    v16 = (const __m128i *)sub_724DC0();
    v14 = v16;
    if ( v8 == 5 )
      v15 = *(_QWORD *)(*(_QWORD *)a2 + 128LL);
    else
      v15 = *(unsigned int *)(*(_QWORD *)a2 + 136LL);
    goto LABEL_23;
  }
  if ( (unsigned int)sub_8DC060(a1) )
  {
    switch ( v8 )
    {
      case 5:
LABEL_26:
        *a4 += 2;
        sub_8238B0(qword_4F18BE0, "st", 2);
        goto LABEL_16;
      case 7:
LABEL_8:
        v9 = *a4;
        v10 = qword_4F18BE0;
        if ( HIDWORD(qword_4F077B4) && unk_4D04250 <= 0x9DCFu )
        {
          if ( a2 )
          {
            *a4 = v9 + 11;
            sub_8238B0(v10, "v18alignofe", 11);
            goto LABEL_14;
          }
          *a4 = v9 + 10;
          sub_8238B0(v10, "v17alignof", 10);
        }
        else
        {
          *a4 = v9 + 2;
          if ( a2 )
          {
            sub_8238B0(v10, "az", 2);
LABEL_14:
            sub_816460(a2, 1u, 0, a4);
            return;
          }
          sub_8238B0(v10, "at", 2);
        }
LABEL_16:
        sub_80F5E0(a1, 0, a4);
        return;
      default:
LABEL_29:
        sub_721090();
    }
  }
  v16 = (const __m128i *)sub_724DC0();
  v14 = v16;
  if ( v8 == 5 )
    v15 = *(_QWORD *)(a1 + 128);
  else
    v15 = *(unsigned int *)(a1 + 136);
LABEL_23:
  sub_72BAF0((__int64)v14, v15, byte_4F06A51[0]);
  sub_80D8A0(v16, 0, 0, a4);
  sub_724E30((__int64)&v16);
}

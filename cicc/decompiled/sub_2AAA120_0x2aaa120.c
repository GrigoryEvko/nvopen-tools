// Function: sub_2AAA120
// Address: 0x2aaa120
//
__int64 __fastcall sub_2AAA120(__int64 a1)
{
  _QWORD *v2; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // edx
  int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // rdi
  _QWORD *v15; // rbx

  LODWORD(v2) = sub_2BFB0D0();
  if ( (_BYTE)v2 )
    return (unsigned int)v2;
  v4 = sub_2BF04A0(a1);
  if ( !v4 || *(_BYTE *)(v4 + 8) != 9 || !a1 )
  {
    v5 = sub_2BF04A0(a1);
    if ( v5 && *(_BYTE *)(v5 + 8) == 17 || (v6 = sub_2BF04A0(a1)) != 0 && *(_BYTE *)(v6 + 8) == 1 )
    {
      v12 = sub_2BF04A0(a1);
      v13 = *(_QWORD **)(v12 + 48);
      v2 = &v13[*(unsigned int *)(v12 + 56)];
      LOBYTE(v2) = v2 == sub_2AA8260(v13, (__int64)v2, (unsigned __int8 (__fastcall *)(_QWORD))sub_2AAA120);
    }
    else
    {
      v7 = sub_2BF04A0(a1);
      if ( !v7 || (LOBYTE(v8) = *(_BYTE *)(v7 + 8) == 4, v9 = v8, LOBYTE(v9) = (a1 != 0) & v8, !(_BYTE)v9) )
      {
        v10 = sub_2BF04A0(a1);
        if ( !v10 || *(_BYTE *)(v10 + 8) != 1 || !a1 )
        {
          v11 = sub_2BF04A0(a1);
          if ( v11 )
            LOBYTE(v2) = *(_BYTE *)(v11 + 8) == 2;
          return (unsigned int)v2;
        }
        goto LABEL_23;
      }
      LODWORD(v2) = a1 - 96;
      if ( (unsigned __int8)sub_2C1A9B0(a1 - 96) || (unsigned __int8)sub_2C1A990(a1 - 96) )
      {
        LODWORD(v2) = v9;
      }
      else
      {
        LOBYTE(v2) = *(_BYTE *)(a1 + 64) == 84 || (unsigned int)*(unsigned __int8 *)(a1 + 64) - 13 <= 0x11;
        if ( (_BYTE)v2 )
        {
LABEL_23:
          v14 = *(_QWORD **)(a1 - 48);
          v15 = &v14[*(unsigned int *)(a1 - 40)];
          LOBYTE(v2) = v15 == sub_2AA8260(v14, (__int64)v15, (unsigned __int8 (__fastcall *)(_QWORD))sub_2AAA120);
        }
      }
    }
    return (unsigned int)v2;
  }
  return *(unsigned __int8 *)(a1 + 64);
}

// Function: sub_13DDBD0
// Address: 0x13ddbd0
//
_QWORD *__fastcall sub_13DDBD0(int a1, unsigned __int8 *a2, unsigned __int8 *a3, _QWORD *a4, unsigned int a5)
{
  _QWORD *v5; // r14
  __int64 v6; // r13
  _QWORD *result; // rax
  __int64 v8; // rdi
  char v9; // al
  unsigned __int8 v10; // al
  unsigned int v11; // [rsp+Ch] [rbp-24h]

  v5 = a4;
  v6 = (__int64)a3;
  switch ( a1 )
  {
    case 11:
      return (_QWORD *)sub_13DE4E0(a2, a3, 0, 0, a4, a5);
    case 12:
      return (_QWORD *)sub_13D69B0(a2, a3, 0, a4);
    case 13:
      return (_QWORD *)sub_13DEB40(a2, a3, 0, 0, a4, a5);
    case 14:
      return (_QWORD *)sub_13D09B0(a2, (__int64)a3, 0, a4);
    case 15:
      return (_QWORD *)sub_13E01C0(a2, a3, a4, a5);
    case 16:
      return (_QWORD *)sub_13D05E0(a2, a3, 0, a4);
    case 17:
      v8 = 17;
      return (_QWORD *)sub_13DD840(v8, a2, a3, a4, a5);
    case 18:
      v11 = a5;
      v9 = sub_14B0710(a2, a3, 1);
      a5 = v11;
      if ( v9 )
        return (_QWORD *)sub_15A04A0(*(_QWORD *)a2);
      a4 = v5;
      a3 = (unsigned __int8 *)v6;
      v8 = 18;
      return (_QWORD *)sub_13DD840(v8, a2, a3, a4, a5);
    case 19:
      return (_QWORD *)sub_13D6CE0(a2, (__int64)a3, 0, a4);
    case 20:
      return (_QWORD *)sub_13E0700(20, a2, a3);
    case 21:
      return (_QWORD *)sub_13E09D0(a2, a3, a4, a5);
    case 22:
      v10 = a2[16];
      if ( v10 > 0x10u )
        goto LABEL_32;
      if ( a3[16] > 0x10u )
        goto LABEL_21;
      result = (_QWORD *)sub_14D6F90(22, a2, a3, *a4);
      if ( result )
        return result;
      v10 = a2[16];
LABEL_21:
      if ( v10 == 9 )
        return (_QWORD *)sub_15A11D0(*(_QWORD *)a2, 0, 0);
LABEL_32:
      if ( *(_BYTE *)(v6 + 16) == 9 )
        return (_QWORD *)sub_15A11D0(*(_QWORD *)a2, 0, 0);
      else
        return (_QWORD *)sub_13CDA40(a2, (_QWORD *)v6);
    case 23:
      result = (_QWORD *)sub_13E0AE0(23, a2, a3);
      if ( !result )
        return (_QWORD *)sub_13D0230(a2, v6, 0, 0);
      return result;
    case 24:
      result = (_QWORD *)sub_13E0EE0(24, a2, a3, 0, a4, a5);
      if ( !result )
        return (_QWORD *)sub_13D7050((__int64)a2, (_BYTE *)v6, v5);
      return result;
    case 25:
      result = (_QWORD *)sub_13E0EE0(25, a2, a3, 0, a4, a5);
      if ( !result )
        return (_QWORD *)sub_13CDFA0(a2, v6, v5);
      return result;
    case 26:
      return (_QWORD *)sub_13DF820(a2, a3, a4, a5);
    case 27:
      return sub_13D7FB0((__int64)a2, (__int64)a3, a4, a5);
    case 28:
      return (_QWORD *)sub_13DE280(a2, a3, a4, a5);
  }
}

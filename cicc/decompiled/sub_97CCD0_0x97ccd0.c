// Function: sub_97CCD0
// Address: 0x97ccd0
//
__int64 __fastcall sub_97CCD0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, _BYTE *a5, __int64 *a6, unsigned int a7)
{
  unsigned int v10; // edi
  __int64 v12; // rsi
  char v13; // al
  __int64 result; // rax
  __int64 v15; // rax
  int v16; // r15d
  _BYTE *v17; // r15
  unsigned __int8 v18; // r8
  int v19; // edx
  __int64 v20; // r13
  int v21; // ecx
  __int64 *v22; // r9
  int v23; // r8d
  int v24; // [rsp+0h] [rbp-B0h]
  int v25; // [rsp+8h] [rbp-A8h]
  int v26; // [rsp+10h] [rbp-A0h]
  int v27; // [rsp+10h] [rbp-A0h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+18h] [rbp-98h]
  int v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v35; // [rsp+28h] [rbp-88h]
  __int64 v36; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v37; // [rsp+38h] [rbp-78h]
  char v38; // [rsp+40h] [rbp-70h]
  __int64 v39; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-58h]
  __int64 v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  char v43; // [rsp+70h] [rbp-40h]

  v10 = a2;
  if ( (_DWORD)a2 == 12 )
    return sub_96E680(12, *a3);
  if ( (unsigned int)(a2 - 13) <= 0x11 )
  {
    v12 = *a3;
    switch ( v10 )
    {
      case 0xEu:
      case 0x10u:
      case 0x12u:
      case 0x15u:
      case 0x18u:
        if ( *(_BYTE *)a1 <= 0x1Cu )
          goto LABEL_10;
        result = sub_96F2E0(v10, v12, (_BYTE *)a3[1], (__int64)a5, a1, a7);
        break;
      default:
LABEL_10:
        result = sub_96E6C0(v10, v12, (_BYTE *)a3[1], (__int64)a5);
        break;
    }
    return result;
  }
  if ( (unsigned int)(a2 - 38) <= 0xC )
    return sub_96F480(a2, *a3, *(_QWORD *)(a1 + 8), (__int64)a5);
  v13 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    if ( v13 != 63 )
    {
      switch ( (int)a2 )
      {
        case ' ':
          if ( v13 != 61 )
            goto LABEL_59;
          if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
            return 0;
          result = sub_9718F0(*a3, *(_QWORD *)(a1 + 8), a5);
          break;
        case '5':
        case '6':
          return sub_9719A0(*(_WORD *)(a1 + 2) & 0x3F, (_BYTE *)*a3, a3[1], (__int64)a5, (__int64)a6, a1);
        case '8':
          goto LABEL_31;
        case '9':
          return sub_AA9CA0(*a3, a3[1], a3[2], a5, a5, a7);
        case '=':
          return sub_AD5840(*a3, a3[1], 0);
        case '>':
          return sub_AD5A90(*a3, a3[1], a3[2], 0);
        case '?':
          return sub_AD5CE0(*a3, a3[1], *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80), 0, a7);
        case '@':
          return sub_AAADB0(*a3, *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80));
        case 'A':
          return sub_AAAE30(*a3, a3[1], *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80));
        case 'C':
          goto LABEL_23;
        default:
          return 0;
      }
      return result;
    }
    goto LABEL_15;
  }
  if ( v13 == 5 )
  {
    if ( *(_WORD *)(a1 + 2) != 34 )
      return sub_ADABF0(a1, a3, a4, *(_QWORD *)(a1 + 8), 0, 0);
LABEL_15:
    v15 = sub_BB5290(a1, a2, a3);
    v16 = v15;
    if ( (unsigned __int8)sub_BCEA30(v15) )
      return 0;
    result = sub_97B6F0(a1, a3, a4, (__int64)a5, (__int64)a6);
    if ( !result )
    {
      sub_BB52D0(&v34, a1);
      v18 = *(_BYTE *)(a1 + 1);
      v19 = (_DWORD)a3 + 8;
      v43 = 0;
      v20 = *a3;
      v21 = a4 - 1;
      v22 = &v39;
      v23 = v18 >> 1;
      if ( v38 )
      {
        v40 = v35;
        if ( v35 > 0x40 )
        {
          v24 = v23;
          v26 = v19;
          sub_C43780(&v39, &v34);
          v23 = v24;
          v21 = a4 - 1;
          v19 = v26;
          v22 = &v39;
        }
        else
        {
          v39 = v34;
        }
        v42 = v37;
        if ( v37 > 0x40 )
        {
          v25 = v23;
          v27 = v21;
          v33 = v19;
          sub_C43780(&v41, &v36);
          LODWORD(v22) = (unsigned int)&v39;
          v23 = v25;
          v21 = v27;
          v19 = v33;
        }
        else
        {
          v41 = v36;
        }
        v43 = 1;
      }
      result = sub_AD9FD0(v16, v20, v19, v21, v23, (_DWORD)v22, 0);
      if ( v43 )
      {
        v43 = 0;
        if ( v42 > 0x40 && v41 )
        {
          v31 = result;
          j_j___libc_free_0_0(v41);
          result = v31;
        }
        if ( v40 > 0x40 && v39 )
        {
          v32 = result;
          j_j___libc_free_0_0(v39);
          result = v32;
        }
      }
      if ( v38 )
      {
        v38 = 0;
        if ( v37 > 0x40 && v36 )
        {
          v29 = result;
          j_j___libc_free_0_0(v36);
          result = v29;
        }
        if ( v35 > 0x40 && v34 )
        {
          v30 = result;
          j_j___libc_free_0_0(v34);
          return v30;
        }
      }
    }
    return result;
  }
  switch ( (int)a2 )
  {
    case ' ':
LABEL_59:
      BUG();
    case '5':
    case '6':
      return sub_9719A0(*(_WORD *)(a1 + 2) & 0x3F, (_BYTE *)*a3, a3[1], (__int64)a5, (__int64)a6, a1);
    case '8':
LABEL_31:
      v17 = (_BYTE *)a3[a4 - 1];
      if ( *v17 || !sub_971E80(a1, (__int64)v17) )
        return 0;
      result = sub_97A150(a1, (__int64)v17, a3, a4 - 1, a6, a7);
      break;
    case '9':
      return sub_AA9CA0(*a3, a3[1], a3[2], a5, a5, a7);
    case '=':
      return sub_AD5840(*a3, a3[1], 0);
    case '>':
      return sub_AD5A90(*a3, a3[1], a3[2], 0);
    case '?':
      return sub_AD5CE0(*a3, a3[1], *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80), 0, a7);
    case '@':
      return sub_AAADB0(*a3, *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80));
    case 'A':
      return sub_AAAE30(*a3, a3[1], *(_QWORD *)(a1 + 72), *(unsigned int *)(a1 + 80));
    case 'C':
LABEL_23:
      if ( !(unsigned __int8)sub_98ED60(*a3, 0, 0, 0, 0, a7) )
        return 0;
      result = *a3;
      break;
    default:
      return 0;
  }
  return result;
}

// Function: sub_B358C0
// Address: 0xb358c0
//
__int64 __fastcall sub_B358C0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8,
        unsigned __int16 a9)
{
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rax
  int v19; // r9d
  __int64 v20; // rsi
  __int64 v21; // r12
  __int64 *v22; // rax
  __int64 v23; // r13
  int v24; // edx
  unsigned int v25; // ecx
  unsigned __int8 v26; // al
  __int64 *v28; // rax
  __int64 v29; // [rsp-10h] [rbp-A0h]
  __int64 v30; // [rsp-8h] [rbp-98h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  __int64 v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  __int64 v39; // [rsp+48h] [rbp-48h]
  __int64 v40; // [rsp+50h] [rbp-40h]

  v11 = (unsigned __int8)a9;
  if ( !HIBYTE(a9) )
    v11 = *(unsigned __int8 *)(a1 + 109);
  sub_E3F8A0(&v38, v11);
  v12 = sub_B9B140(*(_QWORD *)(a1 + 72), v38, v39);
  v13 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v12);
  v14 = *(_DWORD *)(a1 + 104);
  v33 = v13;
  if ( BYTE4(a5) )
    v14 = a5;
  if ( (unsigned __int8)sub_B6B000(a2) )
  {
    v17 = *(unsigned __int8 *)(a1 + 110);
    if ( HIBYTE(a8) )
      v17 = (unsigned __int8)a8;
    sub_E3F6F0(&v38, v17, v15, v16);
    v18 = sub_B9B140(*(_QWORD *)(a1 + 72), v38, v39);
    BYTE4(v35) = 0;
    v39 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v18);
    v19 = 3;
    v38 = a3;
    v40 = v33;
    v36 = a4;
    v30 = a6;
    v37 = *(_QWORD *)(a3 + 8);
    v29 = v35;
  }
  else
  {
    BYTE4(v35) = 0;
    v38 = a3;
    v19 = 2;
    v39 = v33;
    v36 = a4;
    v30 = a6;
    v37 = *(_QWORD *)(a3 + 8);
    v29 = v35;
  }
  v20 = a2;
  v21 = sub_B33D10(a1, a2, (__int64)&v36, 2, (int)&v38, v19, v29, v30);
  v22 = (__int64 *)sub_BD5C60(v21, v20);
  *(_QWORD *)(v21 + 72) = sub_A7A090((__int64 *)(v21 + 72), v22, -1, 72);
  if ( *(_BYTE *)v21 > 0x1Cu )
  {
    switch ( *(_BYTE *)v21 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_14;
      case 'T':
      case 'U':
      case 'V':
        v23 = *(_QWORD *)(v21 + 8);
        v24 = *(unsigned __int8 *)(v23 + 8);
        v25 = v24 - 17;
        v26 = *(_BYTE *)(v23 + 8);
        if ( (unsigned int)(v24 - 17) <= 1 )
          v26 = *(_BYTE *)(**(_QWORD **)(v23 + 16) + 8LL);
        if ( v26 <= 3u || v26 == 5 || (v26 & 0xFD) == 4 )
          goto LABEL_14;
        if ( (_BYTE)v24 == 15 )
        {
          if ( (*(_BYTE *)(v23 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(v21 + 8)) )
            return v21;
          v28 = *(__int64 **)(v23 + 16);
          v23 = *v28;
          v24 = *(unsigned __int8 *)(*v28 + 8);
          v25 = v24 - 17;
        }
        else if ( (_BYTE)v24 == 16 )
        {
          do
          {
            v23 = *(_QWORD *)(v23 + 24);
            LOBYTE(v24) = *(_BYTE *)(v23 + 8);
          }
          while ( (_BYTE)v24 == 16 );
          v25 = (unsigned __int8)v24 - 17;
        }
        if ( v25 <= 1 )
          LOBYTE(v24) = *(_BYTE *)(**(_QWORD **)(v23 + 16) + 8LL);
        if ( (unsigned __int8)v24 <= 3u || (_BYTE)v24 == 5 || (v24 & 0xFD) == 4 )
        {
LABEL_14:
          if ( a7 || (a7 = *(_QWORD *)(a1 + 96)) != 0 )
            sub_B99FD0(v21, 3, a7);
          sub_B45150(v21, v14);
        }
        break;
      default:
        return v21;
    }
  }
  return v21;
}

// Function: sub_E71F00
// Address: 0xe71f00
//
__int64 __fastcall sub_E71F00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 *v8; // r14
  __int64 v9; // r15
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  char v15; // r15
  __int64 v16; // r9
  __int64 v17; // rsi
  unsigned int v18; // r10d
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r15
  void (__fastcall *v22)(_QWORD *, __int64, __int64, _QWORD, __int64, __int64); // rax
  unsigned int v23; // r15d
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // r15d
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  int v31; // r15d
  void (__fastcall *v32)(_QWORD *, __int64, __int64); // rax
  unsigned int v33; // r15d
  unsigned int v34; // r15d
  void (__fastcall *v35)(_QWORD *, __int64, __int64, __int64, __int64, __int64); // rax
  unsigned int v36; // r15d
  __int64 v37; // rsi
  __int64 v38; // rdi
  unsigned int v39; // eax
  int v40; // eax
  unsigned int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rcx
  unsigned int v44; // eax
  unsigned int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // [rsp+8h] [rbp-48h]
  int v48; // [rsp+10h] [rbp-40h]
  unsigned int v49; // [rsp+10h] [rbp-40h]
  unsigned int v50; // [rsp+10h] [rbp-40h]
  unsigned int v51; // [rsp+10h] [rbp-40h]
  int v52; // [rsp+10h] [rbp-40h]
  __int64 v53; // [rsp+18h] [rbp-38h]

  result = (__int64)&a2[13 * a3];
  v53 = result;
  if ( (__int64 *)result != a2 )
  {
    v8 = a2;
    while ( 1 )
    {
      v9 = *v8;
      if ( !*v8 )
        goto LABEL_8;
      if ( *(_QWORD *)v9 )
        break;
      result = *(_BYTE *)(v9 + 9) & 0x70;
      if ( (_BYTE)result == 32 && *(char *)(v9 + 8) >= 0 )
      {
        *(_BYTE *)(v9 + 8) |= 8u;
        result = sub_E807D0(*(_QWORD *)(v9 + 24));
        *(_QWORD *)v9 = result;
        if ( result )
          break;
      }
LABEL_12:
      v8 += 13;
      if ( (__int64 *)v53 == v8 )
        return result;
    }
    if ( a4 && v9 != a4 )
    {
      v10 = a4;
      a4 = v9;
      sub_E8B440(*(_QWORD *)(a1 + 24), v10, v9, v8[5]);
    }
LABEL_8:
    v11 = *(_QWORD **)(a1 + 24);
    v12 = v11[1];
    v13 = *(_QWORD *)(v12 + 152);
    v14 = *(unsigned int *)(v13 + 12);
    if ( !*(_BYTE *)(v13 + 17) )
      v14 = (unsigned int)-(int)v14;
    v15 = *((_BYTE *)v8 + 32);
    v16 = *(_QWORD *)(v12 + 160);
    switch ( v15 )
    {
      case 0:
        v23 = *((_DWORD *)v8 + 2);
        v24 = *v11;
        v25 = 8;
        goto LABEL_34;
      case 1:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
                   v11,
                   10,
                   1,
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 2:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
                   v11,
                   11,
                   1,
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 3:
      case 8:
        v18 = *((_DWORD *)v8 + 2);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v48 = v14;
          v19 = sub_E92290(v16, v18, v13, v14);
          LODWORD(v14) = v48;
          v11 = *(_QWORD **)(a1 + 24);
          v18 = v19;
        }
        v20 = v8[2];
        if ( v15 == 8 )
          v20 -= *(_QWORD *)a1;
        v21 = v20 / (int)v14;
        v22 = *(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64, __int64))(*v11 + 536LL);
        if ( v21 < 0 )
        {
          v51 = v18;
          v22(v11, 17, 1, (int)v14, a5, v16);
          sub_E98EB0(*(_QWORD *)(a1 + 24), v51, 0);
          result = sub_E990E0(*(_QWORD *)(a1 + 24), v21);
        }
        else
        {
          if ( v18 > 0x3F )
          {
            v50 = v18;
            v22(v11, 5, 1, (int)v14, a5, v16);
            sub_E98EB0(*(_QWORD *)(a1 + 24), v50, 0);
          }
          else
          {
            v22(v11, v18 + 128, 1, (int)v14, a5, v16);
          }
          result = sub_E98EB0(*(_QWORD *)(a1 + 24), v21, 0);
        }
        goto LABEL_12;
      case 4:
        v26 = *((_DWORD *)v8 + 2);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v41 = sub_E92290(v16, v26, v13, v14);
          v11 = *(_QWORD **)(a1 + 24);
          v26 = v41;
        }
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
          v11,
          48,
          1,
          v14,
          a5,
          v16);
        sub_E98EB0(*(_QWORD *)(a1 + 24), v26, 0);
        v27 = v8[2];
        v28 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)a1 = v27;
        sub_E98EB0(v28, v27, 0);
        result = sub_E98EB0(*(_QWORD *)(a1 + 24), *((unsigned int *)v8 + 6), 0);
        goto LABEL_12;
      case 5:
        v23 = *((_DWORD *)v8 + 2);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v39 = sub_E92290(v16, v23, v13, v14);
          v11 = *(_QWORD **)(a1 + 24);
          v23 = v39;
        }
        v24 = *v11;
        v25 = 13;
        goto LABEL_34;
      case 6:
      case 9:
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
          v11,
          14,
          1,
          v14,
          a5,
          v16);
        v17 = v8[2];
        if ( v15 == 9 )
          v17 += *(_QWORD *)a1;
        *(_QWORD *)a1 = v17;
        goto LABEL_21;
      case 7:
        v36 = *((_DWORD *)v8 + 2);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v45 = sub_E92290(v16, v36, v13, v14);
          v11 = *(_QWORD **)(a1 + 24);
          v36 = v45;
        }
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
          v11,
          12,
          1,
          v14,
          a5,
          v16);
        sub_E98EB0(*(_QWORD *)(a1 + 24), v36, 0);
        v37 = v8[2];
        v38 = *(_QWORD *)(a1 + 24);
        *(_QWORD *)a1 = v37;
        result = sub_E98EB0(v38, v37, 0);
        goto LABEL_12;
      case 10:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 512LL))(
                   v11,
                   v8[6],
                   v8[7] - v8[6],
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 11:
        v34 = *((_DWORD *)v8 + 2);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v46 = sub_E92290(v16, v34, v13, v14);
          v11 = *(_QWORD **)(a1 + 24);
          v34 = v46;
        }
        v35 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL);
        if ( v34 > 0x3F )
        {
          v35(v11, 6, 1, v14, a5, v16);
          v17 = v34;
LABEL_21:
          result = sub_E98EB0(*(_QWORD *)(a1 + 24), v17, 0);
        }
        else
        {
          result = ((__int64 (__fastcall *)(_QWORD *, _QWORD, __int64, __int64, __int64, __int64))v35)(
                     v11,
                     v34 | 0xC0,
                     1,
                     v14,
                     a5,
                     v16);
        }
        goto LABEL_12;
      case 12:
        v23 = *((_DWORD *)v8 + 2);
        v24 = *v11;
        v25 = 7;
LABEL_34:
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(v24 + 536))(
          v11,
          v25,
          1,
          v14,
          a5,
          v16);
        result = sub_E98EB0(*(_QWORD *)(a1 + 24), v23, 0);
        goto LABEL_12;
      case 13:
        v33 = *((_DWORD *)v8 + 2);
        v49 = *((_DWORD *)v8 + 3);
        if ( !*(_BYTE *)(a1 + 16) )
        {
          v47 = *(_QWORD *)(v12 + 160);
          v33 = sub_E92290(v16, v33, v13, v14);
          v44 = sub_E92290(v47, v49, v42, v43);
          v11 = *(_QWORD **)(a1 + 24);
          v49 = v44;
        }
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
          v11,
          9,
          1,
          v14,
          a5,
          v16);
        sub_E98EB0(*(_QWORD *)(a1 + 24), v33, 0);
        result = sub_E98EB0(*(_QWORD *)(a1 + 24), v49, 0);
        goto LABEL_12;
      case 14:
      case 15:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
                   v11,
                   45,
                   1,
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 16:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
                   v11,
                   44,
                   1,
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 17:
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 536LL))(
          v11,
          46,
          1,
          v14,
          a5,
          v16);
        result = sub_E98EB0(*(_QWORD *)(a1 + 24), v8[2], 0);
        goto LABEL_12;
      case 18:
        result = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v11 + 208LL))(
                   v11,
                   v8[1],
                   v8[5],
                   v14,
                   a5,
                   v16);
        goto LABEL_12;
      case 19:
        v29 = *((unsigned int *)v8 + 2);
        if ( *(_BYTE *)(a1 + 16) )
          goto LABEL_41;
        v52 = v14;
        v40 = sub_E92290(v16, v29, v13, v14);
        LODWORD(v14) = v52;
        LODWORD(v29) = v40;
        if ( *((_BYTE *)v8 + 32) == 4 )
        {
          v30 = v8[2];
          v11 = *(_QWORD **)(a1 + 24);
        }
        else
        {
          v11 = *(_QWORD **)(a1 + 24);
LABEL_41:
          v30 = v8[2];
        }
        v31 = (int)v30 / (int)v14;
        v32 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(*v11 + 536LL);
        if ( v31 < 0 )
        {
          v32(v11, 21, 1);
          sub_E98EB0(*(_QWORD *)(a1 + 24), (unsigned int)v29, 0);
          result = sub_E990E0(*(_QWORD *)(a1 + 24), v31);
        }
        else
        {
          v32(v11, 20, 1);
          sub_E98EB0(*(_QWORD *)(a1 + 24), (unsigned int)v29, 0);
          result = sub_E98EB0(*(_QWORD *)(a1 + 24), v31, 0);
        }
        break;
      default:
        BUG();
    }
    goto LABEL_12;
  }
  return result;
}

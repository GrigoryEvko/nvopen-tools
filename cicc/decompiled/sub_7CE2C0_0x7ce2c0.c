// Function: sub_7CE2C0
// Address: 0x7ce2c0
//
_QWORD *__fastcall sub_7CE2C0(unsigned __int64 a1, _BYTE *a2, char a3, __int64 a4, _DWORD *a5, _QWORD *a6, int a7)
{
  __int64 v8; // rdi
  char v9; // bl
  unsigned int v10; // r15d
  __int64 v11; // r12
  __int64 i; // rax
  unsigned __int64 v13; // rax
  _BOOL4 v14; // r14d
  _BYTE *v15; // rax
  __int64 v16; // r12
  unsigned __int64 v17; // r13
  _QWORD *v18; // rax
  _BYTE *v20; // rax
  _QWORD *v21; // rdx
  bool v22; // dl
  unsigned __int64 v25; // [rsp+18h] [rbp-B8h]
  _BYTE *v26; // [rsp+20h] [rbp-B0h]
  char v27; // [rsp+2Bh] [rbp-A5h]
  unsigned __int8 v28; // [rsp+2Ch] [rbp-A4h]
  int v29; // [rsp+30h] [rbp-A0h]
  int v30; // [rsp+34h] [rbp-9Ch]
  _BYTE *v31; // [rsp+38h] [rbp-98h]
  unsigned __int64 v32; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 v33; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v34; // [rsp+58h] [rbp-78h] BYREF
  unsigned __int64 *v35; // [rsp+60h] [rbp-70h] BYREF
  _QWORD *v36; // [rsp+68h] [rbp-68h]
  int v37; // [rsp+70h] [rbp-60h]
  __int64 v38; // [rsp+78h] [rbp-58h]
  char v39; // [rsp+88h] [rbp-48h]
  bool v40; // [rsp+89h] [rbp-47h]
  char v41; // [rsp+8Ah] [rbp-46h]
  bool v42; // [rsp+8Bh] [rbp-45h]
  char v43; // [rsp+8Ch] [rbp-44h]

  v31 = a2;
  v30 = a3 & 7;
  v33 = a1;
  v8 = a4 + 1;
  switch ( a3 & 7 )
  {
    case 1:
      v9 = 0;
      v27 = 0;
      v28 = 0;
      v10 = 1;
      v25 = 1;
      v11 = ((1LL << (unk_4F06B9C - 1)) - 1) | (1LL << (unk_4F06B9C - 1));
      goto LABEL_8;
    case 2:
      v28 = dword_4D041B4;
      v11 = (1LL << (unk_4F06B9C - 1)) | ((1LL << (unk_4F06B9C - 1)) - 1);
      if ( dword_4D041B4 )
      {
        v9 = 2;
        v27 = 2;
        v10 = 1;
        v25 = 1;
        v28 = a3 & 7;
      }
      else
      {
        v27 = 0;
        v9 = 0;
        v10 = 1;
        v25 = 1;
      }
      goto LABEL_8;
    case 3:
      v27 = 3;
      v9 = 3;
      v10 = unk_4F06B78;
      v28 = a3 & 7;
      goto LABEL_3;
    case 4:
      v27 = 4;
      v9 = 4;
      v10 = unk_4F06B68;
      v28 = a3 & 7;
      goto LABEL_3;
    case 5:
      v27 = 1;
      v9 = 1;
      v28 = 1;
      v10 = unk_4F06B88;
LABEL_3:
      v11 = (1LL << (unk_4F06B9C - 1)) | ((1LL << (unk_4F06B9C - 1)) - 1);
      if ( v10 == 1 )
      {
        v25 = 1;
      }
      else
      {
        v8 *= v10;
        if ( v10 <= 1uLL )
        {
          v25 = v10;
          v10 = 0;
        }
        else
        {
          for ( i = 1; i != v10; ++i )
            v11 |= v11 << dword_4F06BA0;
          v25 = i;
        }
      }
LABEL_8:
      v36 = 0;
      v26 = sub_724830(v8);
      v34 = v26;
      v35 = &v33;
      v37 = 0;
      v38 = 0;
      v39 = 0;
      v40 = ((v30 - 3) & 0xFFFFFFFD) == 0;
      v41 = 0;
      v43 = a7;
      v42 = a7 != 0 && HIDWORD(qword_4F077B4) != 0;
      v29 = a3 & 8;
      if ( (a3 & 8) != 0 )
      {
        v21 = (_QWORD *)unk_4F06458;
        v13 = v33;
        v36 = (_QWORD *)unk_4F06458;
        if ( unk_4F06458 )
        {
          do
          {
            if ( v21[1] >= v33 )
              break;
            v21 = (_QWORD *)*v21;
            v36 = v21;
          }
          while ( v21 );
        }
        v14 = 0;
        v22 = *a2 == 93;
        v29 = v22;
        v31 = &a2[v22];
      }
      else
      {
        v13 = v33;
        v14 = a7 == 0;
      }
      while ( v13 < (unsigned __int64)v31 || v37 > v29 || v36 && *((_DWORD *)v36 + 4) == 1 && v36[1] == v13 )
      {
        if ( v9 == 1 || (unsigned __int8)(v9 - 3) <= 1u )
        {
          sub_7CD760((__int64)&v35, v14, (__int64 *)&v32, v11);
          sub_7CB900(v32, &v34, v10);
        }
        else
        {
          sub_7CD070((__int64)&v35, v14, (__int64 *)&v32, v11, 1, v30 == 2);
          v15 = v34++;
          *v15 = v32;
        }
        v13 = v33;
      }
      if ( v9 == 1 || (unsigned __int8)(v9 - 3) <= 1u )
      {
        v32 = 0;
        sub_7CB900(0, &v34, v10);
      }
      else
      {
        v20 = v34++;
        *v20 = 0;
      }
      v16 = v34 - v26;
      v17 = (v34 - v26) / v25;
      sub_724C70((__int64)xmmword_4F06300, 2);
      v18 = sub_73C8D0(v28, v17);
      *(_QWORD *)word_4F063B0 = v16;
      xmmword_4F06380[0].m128i_i64[0] = (__int64)v18;
      qword_4F063B8 = v26;
      unk_4F063A8 = v27 | unk_4F063A8 & 0xF8;
      *a5 = 0;
      *a6 = 0;
      return a6;
    default:
      sub_721090();
  }
}

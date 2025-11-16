// Function: sub_27266D0
// Address: 0x27266d0
//
unsigned __int16 __fastcall sub_27266D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 *v7; // rax
  _QWORD *v8; // r14
  unsigned __int16 result; // ax
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // r9
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r14
  unsigned __int16 v16; // bx
  signed __int64 v17; // r10
  __int64 v18; // r8
  _BYTE *v19; // r9
  unsigned __int64 v20; // r8
  size_t v21; // r10
  _QWORD *v22; // rdi
  _BYTE *v23; // rdi
  _BYTE *src; // [rsp+8h] [rbp-B8h]
  __int64 srca; // [rsp+8h] [rbp-B8h]
  size_t n; // [rsp+10h] [rbp-B0h]
  size_t na; // [rsp+10h] [rbp-B0h]
  int v28; // [rsp+18h] [rbp-A8h]
  int v29; // [rsp+18h] [rbp-A8h]
  int v30; // [rsp+18h] [rbp-A8h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+28h] [rbp-98h]
  _BYTE *v33; // [rsp+30h] [rbp-90h] BYREF
  __int64 v34; // [rsp+38h] [rbp-88h]
  _BYTE v35[32]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD *v36; // [rsp+60h] [rbp-60h] BYREF
  __int64 v37; // [rsp+68h] [rbp-58h]
  _QWORD v38[10]; // [rsp+70h] [rbp-50h] BYREF

  v7 = sub_DD8400((__int64)a5, a4);
  v8 = sub_DCC810(a5, (__int64)v7, a1, 0, 0);
  if ( sub_D96A50((__int64)v8) )
    return 0;
  v10 = sub_D95540(a3);
  v38[0] = sub_DD2D10((__int64)a5, (__int64)v8, v10);
  v38[1] = a3;
  v36 = v38;
  v37 = 0x200000002LL;
  v11 = sub_DC7EB0(a5, (__int64)&v36, 0, 0);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  result = sub_2726610((__int64)v11, a2, a5);
  if ( !HIBYTE(result) )
  {
    if ( *((_WORD *)v11 + 12) != 8 )
      return 0;
    v13 = (__int64 *)v11[4];
    v14 = v11[5];
    v32 = *v13;
    if ( v14 == 2 )
    {
      v15 = (_QWORD *)v13[1];
      goto LABEL_10;
    }
    v17 = 8 * v14 - 8;
    v31 = v11[6];
    v18 = v17 >> 3;
    v33 = v35;
    v34 = 0x300000000LL;
    if ( (unsigned __int64)v17 > 0x18 )
    {
      srca = 8 * v14 - 8;
      na = (size_t)v13;
      sub_C8D5F0((__int64)&v33, v35, srca >> 3, 8u, v18, v12);
      LODWORD(v18) = srca >> 3;
      v13 = (__int64 *)na;
      v17 = srca;
      v23 = &v33[8 * (unsigned int)v34];
    }
    else
    {
      v19 = v35;
      if ( 8 * v14 == 8 )
        goto LABEL_17;
      v23 = v35;
    }
    v30 = v18;
    memcpy(v23, v13 + 1, v17);
    v19 = v33;
    LODWORD(v17) = v34;
    LODWORD(v18) = v30;
LABEL_17:
    v36 = v38;
    LODWORD(v34) = v17 + v18;
    v20 = (unsigned int)(v17 + v18);
    v37 = 0x400000000LL;
    v21 = 8 * v20;
    if ( v20 > 4 )
    {
      src = v19;
      n = 8 * v20;
      v28 = v20;
      sub_C8D5F0((__int64)&v36, v38, v20, 8u, v20, (__int64)v19);
      LODWORD(v20) = v28;
      v21 = n;
      v19 = src;
      v22 = &v36[(unsigned int)v37];
    }
    else
    {
      if ( !v21 )
      {
LABEL_19:
        LODWORD(v37) = v21 + v20;
        v15 = sub_DBFF60((__int64)a5, (unsigned int *)&v36, v31, 0);
        if ( v36 != v38 )
          _libc_free((unsigned __int64)v36);
        if ( v33 != v35 )
          _libc_free((unsigned __int64)v33);
LABEL_10:
        v16 = sub_2726610(v32, a2, a5);
        result = sub_2726610((__int64)v15, a2, a5);
        if ( HIBYTE(v16) && HIBYTE(result) )
        {
          if ( (unsigned __int8)result >= (unsigned __int8)v16 )
            return v16;
          return result;
        }
        return 0;
      }
      v22 = v38;
    }
    v29 = v20;
    memcpy(v22, v19, v21);
    LODWORD(v21) = v37;
    LODWORD(v20) = v29;
    goto LABEL_19;
  }
  return result;
}

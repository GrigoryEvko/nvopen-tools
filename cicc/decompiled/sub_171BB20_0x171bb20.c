// Function: sub_171BB20
// Address: 0x171bb20
//
__int64 __fastcall sub_171BB20(__int64 a1, char *a2, double a3, double a4, double a5)
{
  char v6; // r15
  __int64 v7; // r13
  void *v8; // rax
  void *v9; // r12
  __int64 *v10; // r12
  __int64 result; // rax
  char v12; // cl
  _BYTE *v13; // rdi
  _BYTE *v14; // rdi
  __int64 v15; // rdi
  char *v16; // rax
  char v17; // al
  float v18; // xmm0_4
  float v19; // xmm0_4
  _BYTE *v20; // rax
  float v22; // xmm0_4
  float v23; // xmm0_4
  float v24; // [rsp+4h] [rbp-8Ch]
  float v25; // [rsp+4h] [rbp-8Ch]
  __int64 v26; // [rsp+8h] [rbp-88h]
  char v27; // [rsp+1Fh] [rbp-71h] BYREF
  _BYTE v28[8]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29[3]; // [rsp+28h] [rbp-68h] BYREF
  _BYTE v30[8]; // [rsp+40h] [rbp-50h] BYREF
  void *v31[9]; // [rsp+48h] [rbp-48h] BYREF

  v6 = *a2;
  if ( !*a2 )
  {
    result = (unsigned int)*((__int16 *)a2 + 1);
    if ( *((_WORD *)a2 + 1) == 1 )
      return result;
    v12 = *(_BYTE *)a1;
    if ( *((_WORD *)a2 + 1) == 0xFFFF )
    {
      if ( v12 )
      {
        v15 = a1 + 16;
        if ( *(void **)(a1 + 16) == sub_16982C0() )
          return sub_169C8D0(v15, a3, a4, a5);
        else
          return sub_1699490(v15);
      }
      else
      {
        *(_WORD *)(a1 + 2) = -*(_WORD *)(a1 + 2);
      }
      return result;
    }
    if ( !v12 )
    {
      result = (unsigned int)(*(__int16 *)(a1 + 2) * (_DWORD)result);
      *(_WORD *)(a1 + 2) = result;
      return result;
    }
    goto LABEL_15;
  }
  if ( *(_BYTE *)a1 )
  {
LABEL_15:
    v7 = *(_QWORD *)(a1 + 16);
    goto LABEL_4;
  }
  v7 = *((_QWORD *)a2 + 2);
  sub_1719070(a1, v7, a3, a4, a5);
  v6 = *a2;
LABEL_4:
  v26 = a1 + 8;
  v8 = sub_16982C0();
  v9 = v8;
  if ( !v6 )
  {
    sub_171A3E0((__int64)v28, v7, *((__int16 *)a2 + 1), a3, a4, a5);
    if ( *(void **)(a1 + 16) == v9 )
    {
      v10 = v29;
      sub_16A1EA0((__int64 *)(a1 + 16), v29, 0, a3, a4, a5);
    }
    else if ( (unsigned __int8)sub_169DE70(v26) || (unsigned __int8)sub_169DE70((__int64)v28) )
    {
      v14 = (_BYTE *)(a1 + 16);
      if ( *(void **)(a1 + 16) == v9 )
      {
        v10 = v29;
        sub_169CAA0((__int64)v14, 0, 0, 0, *(float *)&a3);
      }
      else
      {
        v10 = v29;
        sub_16986F0(v14, 0, 0, 0);
      }
    }
    else if ( *(void **)(a1 + 16) == sub_1698270()
           && ((v16 = (char *)sub_16D40F0((__int64)qword_4FBB490)) == 0 ? (v17 = qword_4FBB490[2]) : (v17 = *v16), v17) )
    {
      v24 = sub_171BAF0((__int64)v28);
      v18 = sub_171BAF0(v26);
      v19 = sub_1C40EA0(&v27, 1, 1, v18, v24);
      if ( (unsigned int)sub_1C40EE0(&v27) )
      {
        v10 = v29;
        sub_169CB40(v26, 0, 0, 0, v19);
      }
      else
      {
        sub_14D1B70((__int64)v30, 1, v19);
        sub_171A510((void **)(a1 + 16), v31);
        v10 = v29;
        sub_127D120(v31);
      }
    }
    else
    {
      v10 = v29;
      sub_169DCA0((__int16 **)(a1 + 16), v29, 0);
    }
    return sub_127D120(v10);
  }
  if ( v8 == *(void **)(a1 + 16) )
    return sub_16A1EA0((__int64 *)(a1 + 16), (__int64 *)a2 + 2, 0, a3, a4, a5);
  if ( !(unsigned __int8)sub_169DE70(v26) && !(unsigned __int8)sub_169DE70((__int64)(a2 + 8)) )
  {
    if ( *(void **)(a1 + 16) != sub_1698270() )
      return sub_169DCA0((__int16 **)(a1 + 16), a2 + 16, 0);
    v20 = sub_16D40F0((__int64)qword_4FBB490);
    if ( !(v20 ? *v20 : LOBYTE(qword_4FBB490[2])) )
      return sub_169DCA0((__int16 **)(a1 + 16), a2 + 16, 0);
    v25 = sub_171BAF0((__int64)(a2 + 8));
    v22 = sub_171BAF0(v26);
    v23 = sub_1C40EA0(v28, 1, 1, v22, v25);
    if ( (unsigned int)sub_1C40EE0(v28) )
      return (__int64)sub_169CB40(v26, 0, 0, 0, v23);
    v10 = (__int64 *)v31;
    sub_14D1B70((__int64)v30, 1, v23);
    sub_171A510((void **)(a1 + 16), v31);
    return sub_127D120(v10);
  }
  v13 = (_BYTE *)(a1 + 16);
  if ( v9 == *(void **)(a1 + 16) )
    return sub_169CAA0((__int64)v13, 0, 0, 0, *(float *)&a3);
  else
    return (__int64)sub_16986F0(v13, 0, 0, 0);
}

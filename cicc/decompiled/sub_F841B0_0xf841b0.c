// Function: sub_F841B0
// Address: 0xf841b0
//
__int64 __fastcall sub_F841B0(__int64 **a1, unsigned __int8 *a2, __int64 a3, unsigned int a4)
{
  unsigned __int8 *v5; // r13
  unsigned int v7; // r15d
  _BOOL8 v8; // rsi
  __int64 *v9; // rax
  unsigned int v10; // ebx
  unsigned __int8 *v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // r13
  __int64 v20; // r15
  unsigned __int8 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  _BYTE *v25; // [rsp+8h] [rbp-88h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  char v27; // [rsp+1Fh] [rbp-71h]
  __int64 v28; // [rsp+28h] [rbp-68h]
  __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30; // [rsp+38h] [rbp-58h]
  _BYTE v31[80]; // [rsp+40h] [rbp-50h] BYREF

  v5 = a2;
  v27 = a4;
  v7 = sub_B19DB0((*a1)[5], (__int64)a2, a3);
  if ( (_BYTE)v7 )
  {
    if ( (_BYTE)a4 )
    {
      sub_F83EF0((__int64)a1, a2);
      sub_B44F30(a2);
      v17 = *a2;
      if ( (unsigned __int8)v17 <= 0x36u )
      {
        v18 = 0x40540000000000LL;
        if ( _bittest64(&v18, v17) )
        {
          v29 = sub_DDD3C0(*a1, a2);
          if ( BYTE4(v29) )
          {
            sub_B447F0(a2, (v29 & 2) != 0);
            sub_B44850(a2, (v29 & 4) != 0);
          }
          return a4;
        }
      }
    }
  }
  else if ( *(_BYTE *)a3 != 84
         && (unsigned __int8)sub_B19720((*a1)[5], *(_QWORD *)(a3 + 40), *((_QWORD *)a2 + 5))
         && (unsigned __int8)sub_F7CE70((*a1)[6], (__int64)a2, a3) )
  {
    v29 = (__int64)v31;
    v30 = 0x400000000LL;
    do
    {
      v8 = (_BOOL8)v5;
      v11 = v5;
      v5 = sub_F7D9C0((__int64)a1, v5, a3, 1);
      if ( !v5 )
      {
        v19 = (_BYTE *)v29;
        goto LABEL_18;
      }
      v14 = (unsigned int)v30;
      v15 = (unsigned int)v30 + 1LL;
      if ( v15 > HIDWORD(v30) )
      {
        sub_C8D5F0((__int64)&v29, v31, v15, 8u, v12, v13);
        v14 = (unsigned int)v30;
      }
      v8 = (_BOOL8)v5;
      *(_QWORD *)(v29 + 8 * v14) = v11;
      v9 = *a1;
      LODWORD(v30) = v30 + 1;
      v10 = sub_B19DB0(v9[5], (__int64)v5, a3);
    }
    while ( !(_BYTE)v10 );
    v19 = (_BYTE *)(v29 + 8LL * (unsigned int)v30);
    v25 = (_BYTE *)v29;
    if ( (_BYTE *)v29 == v19 )
    {
      v7 = v10;
    }
    else
    {
      v20 = a3 + 24;
      do
      {
        v21 = (unsigned __int8 *)*((_QWORD *)v19 - 1);
        sub_F808D0((__int64)a1, (__int64)v21);
        v22 = v26;
        v8 = v20;
        LOWORD(v22) = 0;
        v26 = v22;
        sub_B444E0(v21, v20, v22);
        if ( v27 )
        {
          v8 = (_BOOL8)v21;
          sub_F83EF0((__int64)a1, v21);
          sub_B44F30(v21);
          v23 = *v21;
          if ( (unsigned __int8)v23 <= 0x36u )
          {
            v24 = 0x40540000000000LL;
            if ( _bittest64(&v24, v23) )
            {
              v8 = (_BOOL8)v21;
              v28 = sub_DDD3C0(*a1, v21);
              if ( BYTE4(v28) )
              {
                sub_B447F0(v21, (v28 & 2) != 0);
                v8 = (v28 & 4) != 0;
                sub_B44850(v21, v8);
              }
            }
          }
        }
        v19 -= 8;
      }
      while ( v25 != v19 );
      v19 = (_BYTE *)v29;
      v7 = v10;
    }
LABEL_18:
    if ( v19 != v31 )
      _libc_free(v19, v8);
  }
  return v7;
}

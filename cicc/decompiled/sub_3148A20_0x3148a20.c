// Function: sub_3148A20
// Address: 0x3148a20
//
__int64 __fastcall sub_3148A20(__int64 a1)
{
  _QWORD *v2; // rdi
  __int64 v3; // rdx
  size_t v4; // rcx
  __int64 v5; // rsi
  _QWORD *v6; // rdi
  size_t v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rsi
  size_t v11; // rdx
  size_t v12; // rdx
  _QWORD *v13; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  sub_EA1890((int *)a1);
  *(_BYTE *)a1 = sub_3148790() & 1 | *(_BYTE *)a1 & 0xFE;
  *(_BYTE *)a1 = ((unsigned __int8)sub_31487A0() << 7) | *(_BYTE *)a1 & 0x7F;
  *(_BYTE *)(a1 + 1) = sub_31487B0() & 1 | *(_BYTE *)(a1 + 1) & 0xFE;
  *(_BYTE *)(a1 + 1) = (32 * (sub_31487D0() & 1)) | *(_BYTE *)(a1 + 1) & 0xDF;
  *(_DWORD *)(a1 + 20) = sub_31487C0();
  *(_BYTE *)(a1 + 1) = (4 * (sub_3148800() & 1)) | *(_BYTE *)(a1 + 1) & 0xFB;
  sub_31488A0((__int64 *)&v13);
  v2 = *(_QWORD **)(a1 + 32);
  if ( v13 == src )
  {
    v11 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v2 = src[0];
      else
        memcpy(v2, src, n);
      v11 = n;
      v2 = *(_QWORD **)(a1 + 32);
    }
    *(_QWORD *)(a1 + 40) = v11;
    *((_BYTE *)v2 + v11) = 0;
    v2 = v13;
  }
  else
  {
    v3 = src[0];
    v4 = n;
    if ( v2 == (_QWORD *)(a1 + 48) )
    {
      *(_QWORD *)(a1 + 32) = v13;
      *(_QWORD *)(a1 + 40) = v4;
      *(_QWORD *)(a1 + 48) = v3;
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 48);
      *(_QWORD *)(a1 + 32) = v13;
      *(_QWORD *)(a1 + 40) = v4;
      *(_QWORD *)(a1 + 48) = v3;
      if ( v2 )
      {
        v13 = v2;
        src[0] = v5;
        goto LABEL_5;
      }
    }
    v13 = src;
    v2 = src;
  }
LABEL_5:
  n = 0;
  *(_BYTE *)v2 = 0;
  if ( v13 != src )
    j_j___libc_free_0((unsigned __int64)v13);
  *(_BYTE *)a1 = (4 * (sub_3148810() & 1)) | *(_BYTE *)a1 & 0xFB;
  *(_BYTE *)a1 = (8 * (sub_3148820() & 1)) | *(_BYTE *)a1 & 0xF7;
  *(_BYTE *)a1 = (16 * (sub_3148830() & 1)) | *(_BYTE *)a1 & 0xEF;
  *(_BYTE *)a1 = (32 * (sub_3148840() & 1)) | *(_BYTE *)a1 & 0xDF;
  *(_BYTE *)a1 = ((sub_3148850() & 1) << 6) | *(_BYTE *)a1 & 0xBF;
  *(_BYTE *)(a1 + 2) = sub_3148860();
  *(_BYTE *)(a1 + 3) = sub_3148870();
  *(_BYTE *)(a1 + 4) = sub_3148880();
  *(_BYTE *)(a1 + 5) = sub_3148890();
  *(_DWORD *)(a1 + 16) = sub_31487E0();
  *(_BYTE *)(a1 + 248) = sub_31487F0() & 1 | *(_BYTE *)(a1 + 248) & 0xFE;
  sub_3148960((__int64 *)&v13);
  v6 = *(_QWORD **)(a1 + 128);
  if ( v13 == src )
  {
    v12 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v6 = src[0];
      else
        memcpy(v6, src, n);
      v12 = n;
      v6 = *(_QWORD **)(a1 + 128);
    }
    *(_QWORD *)(a1 + 136) = v12;
    *((_BYTE *)v6 + v12) = 0;
    v6 = v13;
  }
  else
  {
    v7 = n;
    v8 = src[0];
    if ( v6 == (_QWORD *)(a1 + 144) )
    {
      *(_QWORD *)(a1 + 128) = v13;
      *(_QWORD *)(a1 + 136) = v7;
      *(_QWORD *)(a1 + 144) = v8;
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 144);
      *(_QWORD *)(a1 + 128) = v13;
      *(_QWORD *)(a1 + 136) = v7;
      *(_QWORD *)(a1 + 144) = v8;
      if ( v6 )
      {
        v13 = v6;
        src[0] = v9;
        goto LABEL_11;
      }
    }
    v13 = src;
    v6 = src;
  }
LABEL_11:
  n = 0;
  *(_BYTE *)v6 = 0;
  if ( v13 != src )
    j_j___libc_free_0((unsigned __int64)v13);
  return a1;
}

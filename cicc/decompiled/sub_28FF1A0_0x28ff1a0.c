// Function: sub_28FF1A0
// Address: 0x28ff1a0
//
__int64 __fastcall sub_28FF1A0(__int64 a1, __int64 a2, void *a3, void *a4, _BYTE *a5, _BYTE *a6)
{
  __int64 v8; // rdi
  _BYTE *v9; // rax
  void *v10; // rdx
  __int64 v11; // rax
  size_t n; // [rsp+0h] [rbp-80h]
  _BYTE *src; // [rsp+8h] [rbp-78h]
  void *v14; // [rsp+10h] [rbp-70h]
  void *v15; // [rsp+18h] [rbp-68h]
  __int16 v16; // [rsp+30h] [rbp-50h]
  void *v17[4]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v18; // [rsp+60h] [rbp-20h]

  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v15 = a4;
    v16 = 261;
    v14 = a3;
    v17[0] = (void *)sub_BD5D20(a2);
    v17[1] = v10;
    v17[2] = v14;
    v17[3] = v15;
    v18 = 1285;
    sub_CA0F50((__int64 *)a1, v17);
    return a1;
  }
  v8 = a1 + 16;
  if ( a5 )
  {
    *(_QWORD *)a1 = v8;
    v9 = a6;
    v17[0] = a6;
    if ( (unsigned __int64)a6 > 0xF )
    {
      n = (size_t)a6;
      src = a5;
      v11 = sub_22409D0(a1, (unsigned __int64 *)v17, 0);
      a5 = src;
      a6 = (_BYTE *)n;
      *(_QWORD *)a1 = v11;
      v8 = v11;
      *(void **)(a1 + 16) = v17[0];
    }
    else
    {
      if ( a6 == (_BYTE *)1 )
      {
        *(_BYTE *)(a1 + 16) = *a5;
LABEL_7:
        *(_QWORD *)(a1 + 8) = v9;
        v9[v8] = 0;
        return a1;
      }
      if ( !a6 )
        goto LABEL_7;
    }
    memcpy((void *)v8, a5, (size_t)a6);
    v9 = v17[0];
    v8 = *(_QWORD *)a1;
    goto LABEL_7;
  }
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}

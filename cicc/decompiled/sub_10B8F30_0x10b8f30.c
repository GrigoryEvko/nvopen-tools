// Function: sub_10B8F30
// Address: 0x10b8f30
//
__int64 __fastcall sub_10B8F30(__int64 **a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v3; // rbp
  __int64 *v4; // rax
  __int64 *v5; // rcx
  _BYTE *v7; // r8
  _BYTE *v8; // r8
  __int64 v9; // r9
  __int64 v10; // r8
  _BYTE *v11; // rsi
  __int64 *v12; // rcx
  __int64 *v13; // rax
  _BYTE *v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // r9
  __int64 v18; // r8
  _WORD v19[24]; // [rsp-38h] [rbp-38h] BYREF
  __int64 v20; // [rsp-8h] [rbp-8h]

  v4 = *a1;
  v5 = a1[1];
  if ( *a2 != 59 )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v7 == 57 && (v17 = *((_QWORD *)v7 - 8)) != 0 && (*v4 = v17, (v18 = *((_QWORD *)v7 - 4)) != 0) )
  {
    *v5 = v18;
    v8 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( v8 == (_BYTE *)*v4 )
      goto LABEL_9;
  }
  else
  {
    v8 = (_BYTE *)*((_QWORD *)a2 - 4);
  }
  if ( *v8 != 57 )
    return 0;
  v9 = *((_QWORD *)v8 - 8);
  if ( !v9 )
    return 0;
  *v4 = v9;
  v10 = *((_QWORD *)v8 - 4);
  if ( !v10 )
    return 0;
  *v5 = v10;
  if ( *((_QWORD *)a2 - 8) != *v4 )
    return 0;
LABEL_9:
  if ( *a3 != 59 )
    return 0;
  v11 = (_BYTE *)*((_QWORD *)a3 - 8);
  v12 = a1[1];
  v13 = *a1;
  v14 = (_BYTE *)*((_QWORD *)a3 - 4);
  v15 = *v12;
  v16 = **a1;
  if ( (*v11 != 57 || v16 != *((_QWORD *)v11 - 8) || v15 != *((_QWORD *)v11 - 4) || (_BYTE *)v15 != v14)
    && (*v14 != 57 || v16 != *((_QWORD *)v14 - 8) || *((_QWORD *)v14 - 4) != v15 || (_BYTE *)v15 != v11) )
  {
    return 0;
  }
  v20 = v3;
  v19[16] = 257;
  return sub_B504D0(30, *v13, *v12, (__int64)v19, 0, 0);
}

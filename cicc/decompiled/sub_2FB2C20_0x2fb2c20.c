// Function: sub_2FB2C20
// Address: 0x2fb2c20
//
__int64 __fastcall sub_2FB2C20(_QWORD *a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  unsigned __int64 v7; // rax
  __int64 v10; // r13
  __int16 v11; // ax
  __int64 *v12; // rsi
  __int64 v13; // rdx
  __int64 (__fastcall *v14)(__int64); // rcx
  _BYTE v15[72]; // [rsp-48h] [rbp-48h] BYREF

  v6 = 1;
  v7 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 != (*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL) )
    return 1;
  if ( a3 )
  {
    if ( *(_BYTE *)(a2 + 32) )
    {
      v6 = *(unsigned __int8 *)(a2 + 33);
      if ( (_BYTE)v6 )
        return (unsigned int)v6;
    }
    v10 = *(_QWORD *)(v7 + 16);
    v11 = *(_WORD *)(v10 + 68);
    if ( v11 != 20 )
    {
      v12 = (__int64 *)a1[4];
      v13 = *v12;
      v14 = *(__int64 (__fastcall **)(__int64))(*v12 + 520);
      if ( v14 == sub_2DCA430 )
        goto LABEL_8;
      ((void (__fastcall *)(_BYTE *, __int64 *, __int64, __int64 (__fastcall *)(__int64), __int64))v14)(
        v15,
        v12,
        v10,
        v14,
        v6);
      if ( !v15[16] )
      {
        v11 = *(_WORD *)(v10 + 68);
LABEL_8:
        if ( v11 != 12 )
          return sub_2FB1CD0(a1, *(_QWORD *)(a2 + 8), v13, (__int64)v14, v6, a6);
      }
    }
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  return 0;
}

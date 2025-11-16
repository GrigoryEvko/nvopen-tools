// Function: sub_C65B40
// Address: 0xc65b40
//
_QWORD *__fastcall sub_C65B40(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  _QWORD *v7; // rdi
  __int64 v8; // rsi
  unsigned int v9; // r13d
  _QWORD *v10; // r12
  _BYTE *v11; // rdi
  __int64 v14; // [rsp+8h] [rbp-D8h]
  _BYTE *v16; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-B8h]
  _BYTE v18[176]; // [rsp+30h] [rbp-B0h] BYREF

  v7 = *(_QWORD **)a2;
  v8 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  v9 = sub_939680(v7, v8);
  v10 = *(_QWORD **)(*(_QWORD *)a1 + 8LL * (v9 & (*(_DWORD *)(a1 + 8) - 1)));
  v14 = *(_QWORD *)a1 + 8LL * (v9 & (*(_DWORD *)(a1 + 8) - 1));
  v16 = v18;
  *a3 = 0;
  v17 = 0x2000000000LL;
  if ( !v10 || ((unsigned __int8)v10 & 1) != 0 )
  {
    v11 = v18;
LABEL_7:
    v10 = 0;
    *a3 = v14;
  }
  else
  {
    while ( 1 )
    {
      v8 = (__int64)v10;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, _BYTE **))(a4 + 8))(
             a1,
             v10,
             a2,
             v9,
             &v16) )
      {
        break;
      }
      LODWORD(v17) = 0;
      v10 = (_QWORD *)*v10;
      if ( !v10 || ((unsigned __int8)v10 & 1) != 0 )
      {
        v11 = v16;
        goto LABEL_7;
      }
    }
    v11 = v16;
  }
  if ( v11 != v18 )
    _libc_free(v11, v8);
  return v10;
}

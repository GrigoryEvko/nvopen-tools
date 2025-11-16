// Function: sub_16BDDE0
// Address: 0x16bdde0
//
_QWORD *__fastcall sub_16BDDE0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // r15d
  _QWORD *v5; // r12
  _BYTE *v6; // rdi
  __int64 v9; // [rsp+10h] [rbp-D0h]
  _BYTE *v10; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v11; // [rsp+28h] [rbp-B8h]
  _BYTE v12[176]; // [rsp+30h] [rbp-B0h] BYREF

  v4 = sub_16BDDB0(a2);
  v5 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 8LL * (v4 & (*(_DWORD *)(a1 + 16) - 1)));
  v9 = *(_QWORD *)(a1 + 8) + 8LL * (v4 & (*(_DWORD *)(a1 + 16) - 1));
  v10 = v12;
  *a3 = 0;
  v11 = 0x2000000000LL;
  if ( !v5 || ((unsigned __int8)v5 & 1) != 0 )
  {
    v6 = v12;
LABEL_7:
    v5 = 0;
    *a3 = v9;
  }
  else
  {
    while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, _BYTE **))(*(_QWORD *)a1 + 16LL))(
               a1,
               v5,
               a2,
               v4,
               &v10) )
    {
      LODWORD(v11) = 0;
      v5 = (_QWORD *)*v5;
      if ( !v5 || ((unsigned __int8)v5 & 1) != 0 )
      {
        v6 = v10;
        goto LABEL_7;
      }
    }
    v6 = v10;
  }
  if ( v6 != v12 )
    _libc_free((unsigned __int64)v6);
  return v5;
}

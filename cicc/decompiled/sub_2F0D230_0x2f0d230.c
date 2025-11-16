// Function: sub_2F0D230
// Address: 0x2f0d230
//
__int64 __fastcall sub_2F0D230(__int64 a1, __int64 a2, _BYTE *a3, _WORD *a4, unsigned __int8 a5)
{
  __int64 v7; // rax
  unsigned __int8 v8; // bl
  __int64 result; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  char v15; // [rsp+17h] [rbp-49h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h] BYREF
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v18; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)a1;
  v15 = 1;
  v8 = (*(__int64 (__fastcall **)(__int64))(v7 + 16))(a1);
  if ( v8 )
    v8 = a3[1] ^ 1;
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( (_BYTE)result )
  {
    if ( !a3[1] )
      goto LABEL_5;
  }
  else if ( !a3[1] )
  {
    *(_WORD *)a3 = 256;
  }
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             a2,
             a5,
             v8,
             &v15,
             &v16);
  if ( !(_BYTE)result )
  {
LABEL_5:
    if ( v15 )
    {
      result = (unsigned __int16)*a4;
      *(_WORD *)a3 = result;
    }
    return result;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    goto LABEL_21;
  v10 = sub_CB1000(a1);
  if ( *(_DWORD *)(v10 + 32) != 1 )
    goto LABEL_21;
  v11 = *(_QWORD *)(v10 + 80);
  v17 = *(_QWORD *)(v10 + 72);
  v18 = v11;
  v12 = sub_C93710(&v17, 32, 0xFFFFFFFFFFFFFFFFLL) + 1;
  if ( v12 > v18 )
    v12 = v18;
  v13 = v18 - v11 + v12;
  if ( v13 > v18 )
    v13 = v18;
  if ( v13 == 6 && *(_DWORD *)v17 == 1852796476 && *(_WORD *)(v17 + 4) == 15973 )
    *(_WORD *)a3 = *a4;
  else
LABEL_21:
    sub_2F07940(a1, a3);
  return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v16);
}

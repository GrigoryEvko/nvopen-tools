// Function: sub_710280
// Address: 0x710280
//
_BOOL8 __fastcall sub_710280(const __m128i *a1, unsigned __int8 a2, __m128i *a3, int a4, _DWORD *a5)
{
  char *v7; // rdi
  _BOOL8 result; // rax
  int v9; // ecx
  int v10; // [rsp+4h] [rbp-5Ch]
  int v12; // [rsp+1Ch] [rbp-44h] BYREF
  int v13; // [rsp+20h] [rbp-40h] BYREF
  int v14; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v12 = 0;
  v10 = sub_70C860(a2, a1);
  if ( !(a4 | v10) )
  {
    sub_70B830(a2, a1, v15, &v12, a5);
    sub_620DE0(a3, v15[0]);
    v9 = v12;
    goto LABEL_9;
  }
  sub_70B720(a2, a1, v15, &v12, a5);
  sub_620D80(a3, v15[0]);
  if ( a4 )
  {
    v9 = v12;
LABEL_9:
    result = 1;
    if ( !v9 )
      return result;
    goto LABEL_4;
  }
  v12 = 1;
LABEL_4:
  v7 = sub_70B160(a2, a1, &v13, &v14, v15);
  if ( LODWORD(v15[0]) | v14 | v13 )
    return 0;
  if ( a4 || !v10 )
  {
    sub_622150(v7, a3, a4, &v12);
    return v12 == 0;
  }
  else
  {
    sub_622150(v7, a3, 1, &v12);
    return 0;
  }
}

// Function: sub_277C800
// Address: 0x277c800
//
unsigned __int64 __fastcall sub_277C800(__int64 *a1)
{
  unsigned __int8 *v1; // rbx
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 *v5; // rsi
  __int64 *v6; // rdi
  int v7; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v8[2]; // [rsp+8h] [rbp-18h] BYREF

  v1 = (unsigned __int8 *)*a1;
  v2 = *(_DWORD *)(*a1 + 4) & 0x7FFFFFF;
  if ( *((_BYTE *)a1 + 16) )
  {
    v8[0] = *(_QWORD *)&v1[-32 * v2];
    v7 = *v1 - 29;
    return sub_277BE00(&v7, v8, a1 + 1);
  }
  else
  {
    v4 = 4 * v2;
    if ( (v1[7] & 0x40) != 0 )
    {
      v6 = (__int64 *)*((_QWORD *)v1 - 1);
      v5 = &v6[v4];
    }
    else
    {
      v5 = (__int64 *)*a1;
      v6 = (__int64 *)&v1[-(v4 * 8)];
    }
    v8[0] = sub_F58E90(v6, v5);
    v7 = *v1 - 29;
    return sub_C4ECF0(&v7, v8);
  }
}

// Function: sub_3704CB0
// Address: 0x3704cb0
//
unsigned __int64 *__fastcall sub_3704CB0(unsigned __int64 *a1, __int16 *a2, __int64 *a3)
{
  __int16 v3; // ax
  __int64 v4; // rax
  __int64 v6; // [rsp+8h] [rbp-38h] BYREF
  __int16 v7; // [rsp+10h] [rbp-30h] BYREF
  __int64 v8; // [rsp+12h] [rbp-2Eh]
  __int16 v9; // [rsp+1Ah] [rbp-26h]
  __int64 v10; // [rsp+20h] [rbp-20h]
  __int64 v11; // [rsp+28h] [rbp-18h]

  v3 = *a2;
  v8 = 0;
  v10 = 0;
  v7 = v3;
  v9 = 0;
  v4 = *a3;
  v11 = 0;
  (*(void (__fastcall **)(__int64 *, __int64 *, __int16 *, __int16 *))(v4 + 200))(&v6, a3, a2, &v7);
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v6 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v6 = 0;
    sub_9C66B0(&v6);
    *a1 = 1;
  }
  return a1;
}

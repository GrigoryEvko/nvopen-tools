// Function: sub_37056D0
// Address: 0x37056d0
//
unsigned __int64 *__fastcall sub_37056D0(unsigned __int64 *a1, __int16 *a2, __int64 a3)
{
  __int16 v3; // ax
  __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  _WORD v6[2]; // [rsp+10h] [rbp-20h] BYREF
  int v7; // [rsp+14h] [rbp-1Ch]
  __int64 v8; // [rsp+18h] [rbp-18h]

  v3 = *a2;
  v7 = 0;
  v8 = 0;
  v6[0] = v3;
  v6[1] = 0;
  (*(void (__fastcall **)(__int64 *, __int64, __int16 *, _WORD *))(*(_QWORD *)a3 + 192LL))(&v5, a3, a2, v6);
  if ( (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v5 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    *a1 = 1;
    v5 = 0;
    sub_9C66B0(&v5);
  }
  return a1;
}

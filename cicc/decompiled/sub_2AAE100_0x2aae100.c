// Function: sub_2AAE100
// Address: 0x2aae100
//
unsigned __int8 *__fastcall sub_2AAE100(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, char a6, char a7)
{
  unsigned __int8 *v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rdx
  unsigned int v14; // esi
  _BYTE v16[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v17; // [rsp+20h] [rbp-40h]

  v17 = 257;
  v10 = (unsigned __int8 *)sub_B504D0(a2, a3, a4, (__int64)v16, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v10,
    a5,
    a1[7],
    a1[8]);
  v11 = *a1;
  v12 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v12 )
  {
    do
    {
      v13 = *(_QWORD *)(v11 + 8);
      v14 = *(_DWORD *)v11;
      v11 += 16;
      sub_B99FD0((__int64)v10, v14, v13);
    }
    while ( v12 != v11 );
  }
  if ( a6 )
  {
    sub_B447F0(v10, 1);
    if ( !a7 )
      return v10;
LABEL_7:
    sub_B44850(v10, 1);
    return v10;
  }
  if ( a7 )
    goto LABEL_7;
  return v10;
}

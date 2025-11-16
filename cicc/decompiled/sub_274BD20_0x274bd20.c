// Function: sub_274BD20
// Address: 0x274bd20
//
unsigned __int64 __fastcall sub_274BD20(__int64 *a1, unsigned __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v6; // r12
  int v7; // ebx
  bool v8; // zf
  int v9; // eax
  __int64 v10; // rdi
  unsigned int v11; // ebx
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // r12
  __int64 v15; // rdx
  int v16; // r13d
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  char v22[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = sub_BCB060(v6);
  v8 = v7 == (unsigned int)sub_BCB060((__int64)a3);
  v9 = 38;
  if ( v8 )
    v9 = 49;
  if ( (__int64 **)v6 == a3 )
    return a2;
  v10 = a1[10];
  v11 = v9;
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v12 != sub_920130 )
  {
    v13 = v12(v10, v11, (_BYTE *)a2, (__int64)a3);
    goto LABEL_8;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(v11) )
      v13 = sub_ADAB70(v11, a2, a3, 0);
    else
      v13 = sub_AA93C0(v11, a2, (__int64)a3);
LABEL_8:
    if ( v13 )
      return v13;
  }
  v23 = 257;
  v13 = sub_B51D30(v11, a2, (__int64)a3, (__int64)v22, 0, 0);
  if ( (unsigned __int8)sub_920620(v13) )
  {
    v15 = a1[12];
    v16 = *((_DWORD *)a1 + 26);
    if ( v15 )
      sub_B99FD0(v13, 3u, v15);
    sub_B45150(v13, v16);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v13,
    a4,
    a1[7],
    a1[8]);
  v17 = *a1;
  v18 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v18 )
  {
    do
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v13, v20, v19);
    }
    while ( v18 != v17 );
  }
  return v13;
}

// Function: sub_10A15A0
// Address: 0x10a15a0
//
unsigned __int8 *__fastcall sub_10A15A0(__int64 a1, _BYTE *a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v6; // rax
  unsigned __int8 *result; // rax
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rdi
  unsigned __int8 *v11; // r14
  __int64 v12; // rdx
  int v13; // r13d
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rdx
  unsigned int v17; // esi
  int v18; // [rsp+4h] [rbp-ACh]
  __int64 v19; // [rsp+8h] [rbp-A8h]
  char v20; // [rsp+10h] [rbp-A0h]
  char v21; // [rsp+14h] [rbp-9Ch]
  __int64 v22; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v23; // [rsp+18h] [rbp-98h]
  _BYTE v24[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v25; // [rsp+40h] [rbp-70h]
  _BYTE v26[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v27; // [rsp+70h] [rbp-40h]

  v3 = *((_QWORD *)a2 - 8);
  v4 = *((_QWORD *)a2 - 4);
  if ( (unsigned __int8)(*(_BYTE *)v3 - 42) >= 0x12u )
    v3 = 0;
  if ( (unsigned __int8)(*(_BYTE *)v4 - 42) > 0x11u )
  {
    sub_F0C3E0(a1, a2);
    return 0;
  }
  if ( sub_F0C3E0(a1, a2) || !v3 )
    return 0;
  v6 = *(_QWORD *)(v3 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
  {
    result = *(unsigned __int8 **)(v4 + 16);
    if ( !result )
      return result;
    if ( *((_QWORD *)result + 1) )
      return 0;
  }
  if ( *(_BYTE *)v3 != 54 )
    return 0;
  v8 = *(_QWORD *)(v3 - 64);
  if ( !v8 )
    return 0;
  v22 = *(_QWORD *)(v3 - 32);
  if ( !v22 )
    return 0;
  if ( *(_BYTE *)v4 != 54 )
    return 0;
  v19 = *(_QWORD *)(v4 - 64);
  if ( !v19 || v22 != *(_QWORD *)(v4 - 32) )
    return 0;
  if ( sub_B44900((__int64)a2) )
  {
    v20 = 0;
    if ( sub_B44900(v3) )
      v20 = sub_B44900(v4);
  }
  else
  {
    v20 = 0;
  }
  v21 = 0;
  if ( sub_B448F0((__int64)a2) && sub_B448F0(v3) )
    v21 = sub_B448F0(v4);
  v9 = (unsigned __int8)*a2;
  v10 = a3[10];
  v25 = 257;
  v18 = v9 - 29;
  v11 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v10 + 16LL))(
                             v10,
                             (unsigned int)(v9 - 29),
                             v8,
                             v19);
  if ( !v11 )
  {
    v27 = 257;
    v11 = (unsigned __int8 *)sub_B504D0(v18, v8, v19, (__int64)v26, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v11) )
    {
      v12 = a3[12];
      v13 = *((_DWORD *)a3 + 26);
      if ( v12 )
        sub_B99FD0((__int64)v11, 3u, v12);
      sub_B45150((__int64)v11, v13);
    }
    (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
      a3[11],
      v11,
      v24,
      a3[7],
      a3[8]);
    v14 = *a3;
    v15 = *a3 + 16LL * *((unsigned int *)a3 + 2);
    while ( v15 != v14 )
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0((__int64)v11, v17, v16);
    }
  }
  if ( (unsigned __int8)(*v11 - 42) <= 0x11u )
  {
    sub_B44850(v11, v20);
    sub_B447F0(v11, v21);
  }
  v27 = 257;
  v23 = (unsigned __int8 *)sub_B504D0(25, (__int64)v11, v22, (__int64)v26, 0, 0);
  sub_B44850(v23, v20);
  sub_B447F0(v23, v21);
  return v23;
}

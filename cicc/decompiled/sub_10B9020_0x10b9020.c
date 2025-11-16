// Function: sub_10B9020
// Address: 0x10b9020
//
__int64 __fastcall sub_10B9020(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 *v6; // r12
  int v7; // edx
  __int64 v8; // r14
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // [rsp+8h] [rbp-98h]
  _BYTE v21[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v22; // [rsp+30h] [rbp-70h]
  _BYTE v23[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v24; // [rsp+60h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  if ( (_DWORD)v3 )
  {
    v22 = 257;
    v10 = sub_AD64C0(*(_QWORD *)(v4 + 8), v3, 0);
    v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a2[10] + 24LL))(
            a2[10],
            26,
            v4,
            v10,
            0);
    if ( !v11 )
    {
      v24 = 257;
      v11 = sub_B504D0(26, v4, v10, (__int64)v23, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v11,
        v21,
        a2[7],
        a2[8]);
      v12 = *a2;
      v13 = *a2 + 16LL * *((unsigned int *)a2 + 2);
      if ( *a2 != v13 )
      {
        do
        {
          v14 = *(_QWORD *)(v12 + 8);
          v15 = *(_DWORD *)v12;
          v12 += 16;
          sub_B99FD0(v11, v15, v14);
        }
        while ( v13 != v12 );
      }
    }
    v4 = v11;
  }
  v5 = *(_QWORD *)(v4 + 8);
  v6 = (__int64 *)sub_BCD140(*(_QWORD **)v5, *(_DWORD *)(a1 + 12));
  v7 = *(unsigned __int8 *)(v5 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
  {
    BYTE4(v20) = (_BYTE)v7 == 18;
    LODWORD(v20) = *(_DWORD *)(v5 + 32);
    v6 = (__int64 *)sub_BCE1B0(v6, v20);
  }
  if ( *(__int64 **)(v4 + 8) != v6 )
  {
    v22 = 257;
    if ( v6 == *(__int64 **)(v4 + 8) )
      return v4;
    v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)a2[10] + 120LL))(
           a2[10],
           38,
           v4,
           v6);
    if ( !v8 )
    {
      v24 = 257;
      v8 = sub_B51D30(38, v4, (__int64)v6, (__int64)v23, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v8,
        v21,
        a2[7],
        a2[8]);
      v16 = *a2;
      v17 = *a2 + 16LL * *((unsigned int *)a2 + 2);
      while ( v17 != v16 )
      {
        v18 = *(_QWORD *)(v16 + 8);
        v19 = *(_DWORD *)v16;
        v16 += 16;
        sub_B99FD0(v8, v19, v18);
      }
    }
    return v8;
  }
  return v4;
}

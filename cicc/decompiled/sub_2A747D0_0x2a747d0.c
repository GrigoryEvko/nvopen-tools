// Function: sub_2A747D0
// Address: 0x2a747d0
//
__int64 __fastcall sub_2A747D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rbx
  __int64 v5; // rcx
  int v6; // esi
  __int64 v7; // r14
  __int64 v8; // r12
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // rdx
  int v16; // r13d
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // esi
  _BYTE v21[32]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v22; // [rsp+20h] [rbp-60h]
  _BYTE v23[32]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v24; // [rsp+50h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  if ( **(_QWORD **)a1 == a2 )
    return *(_QWORD *)v2;
  v3 = *(__int64 **)(a1 + 24);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = **(_DWORD **)(a1 + 16);
  v22 = 257;
  v7 = *(_QWORD *)(*(_QWORD *)v2 + 8LL);
  if ( v6 )
  {
    if ( v7 != v5 )
    {
      v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v3[10] + 120LL))(
             v3[10],
             40,
             a2,
             *(_QWORD *)(*(_QWORD *)v2 + 8LL));
      if ( !v8 )
      {
        v24 = 257;
        v8 = sub_B51D30(40, a2, v7, (__int64)v23, 0, 0);
        if ( (unsigned __int8)sub_920620(v8) )
        {
          v15 = v3[12];
          v16 = *((_DWORD *)v3 + 26);
          if ( v15 )
            sub_B99FD0(v8, 3u, v15);
          sub_B45150(v8, v16);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v3[11] + 16LL))(
          v3[11],
          v8,
          v21,
          v3[7],
          v3[8]);
        v17 = *v3;
        v18 = *v3 + 16LL * *((unsigned int *)v3 + 2);
        while ( v18 != v17 )
        {
          v19 = *(_QWORD *)(v17 + 8);
          v20 = *(_DWORD *)v17;
          v17 += 16;
          sub_B99FD0(v8, v20, v19);
        }
      }
      return v8;
    }
  }
  else if ( v7 != v5 )
  {
    v8 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v3[10] + 120LL))(
           v3[10],
           39,
           a2,
           *(_QWORD *)(*(_QWORD *)v2 + 8LL));
    if ( !v8 )
    {
      v24 = 257;
      v10 = sub_BD2C40(72, 1u);
      v8 = (__int64)v10;
      if ( v10 )
        sub_B515B0((__int64)v10, a2, v7, (__int64)v23, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v3[11] + 16LL))(
        v3[11],
        v8,
        v21,
        v3[7],
        v3[8]);
      v11 = *v3;
      v12 = *v3 + 16LL * *((unsigned int *)v3 + 2);
      while ( v12 != v11 )
      {
        v13 = *(_QWORD *)(v11 + 8);
        v14 = *(_DWORD *)v11;
        v11 += 16;
        sub_B99FD0(v8, v14, v13);
      }
    }
    return v8;
  }
  return a2;
}

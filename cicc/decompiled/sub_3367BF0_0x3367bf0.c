// Function: sub_3367BF0
// Address: 0x3367bf0
//
__int64 __fastcall sub_3367BF0(__int64 a1, __int64 a2, __int128 *a3)
{
  __int64 v4; // r13
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int); // r15
  __int64 v6; // rax
  int v7; // edx
  __int16 v8; // ax
  __int64 *v9; // rdi
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int); // r15
  __int64 v11; // rax
  unsigned __int16 v12; // r14
  int v13; // eax
  _QWORD *v14; // r15
  int v15; // r9d
  __int64 v16; // r13
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v23; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v25; // [rsp+10h] [rbp-B0h]
  unsigned int v26; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-98h]
  __int64 v28; // [rsp+30h] [rbp-90h] BYREF
  char v29; // [rsp+38h] [rbp-88h]
  __int64 v30; // [rsp+40h] [rbp-80h]
  __int64 v31; // [rsp+48h] [rbp-78h]
  __int128 v32; // [rsp+50h] [rbp-70h]
  __int64 v33; // [rsp+60h] [rbp-60h]
  _QWORD v34[10]; // [rsp+70h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 32LL);
  v6 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v5 == sub_2D42F30 )
  {
    v7 = sub_AE2980(v6, 0)[1];
    v8 = 2;
    if ( v7 != 1 )
    {
      v8 = 3;
      if ( v7 != 2 )
      {
        v8 = 4;
        if ( v7 != 4 )
        {
          v8 = 5;
          if ( v7 != 8 )
          {
            v8 = 6;
            if ( v7 != 16 )
            {
              v8 = 7;
              if ( v7 != 32 )
              {
                v8 = 8;
                if ( v7 != 64 )
                  v8 = 9 * (v7 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v8 = v5(v4, v6, 0);
  }
  LOWORD(v26) = v8;
  v9 = *(__int64 **)(a1 + 40);
  v27 = 0;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v4 + 40LL);
  v11 = sub_2E79000(v9);
  if ( v10 == sub_2D42FA0 )
  {
    v12 = 2;
    v13 = sub_AE2980(v11, 0)[1];
    if ( v13 != 1 )
    {
      v12 = 3;
      if ( v13 != 2 )
      {
        v12 = 4;
        if ( v13 != 4 )
        {
          v12 = 5;
          if ( v13 != 8 )
          {
            v12 = 6;
            if ( v13 != 16 )
            {
              v12 = 7;
              if ( v13 != 32 )
              {
                v12 = 8;
                if ( v13 != 64 )
                  v12 = 9 * (v13 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v12 = v10(v4, v11, 0);
  }
  v14 = *(_QWORD **)(a1 + 40);
  v23 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 936LL))(v4, *(_QWORD *)(*v14 + 40LL));
  v16 = sub_33F7740(a1, 29, a2, v26, v27, v15, *a3);
  if ( v23 )
  {
    v17 = *(_QWORD *)(v23 + 8);
    *((_QWORD *)&v32 + 1) = 0;
    BYTE4(v33) = 0;
    *(_QWORD *)&v32 = v23 & 0xFFFFFFFFFFFFFFFBLL;
    if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
      v17 = **(_QWORD **)(v17 + 16);
    v18 = *(_DWORD *)(v17 + 8);
    memset(v34, 0, 32);
    LODWORD(v33) = v18 >> 8;
    v25 = sub_33CC4A0(a1, v26, v27);
    if ( (_WORD)v26 )
    {
      if ( (_WORD)v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
        BUG();
      v20 = 16LL * ((unsigned __int16)v26 - 1);
      v19 = *(_QWORD *)&byte_444C4A0[v20];
      LOBYTE(v20) = byte_444C4A0[v20 + 8];
    }
    else
    {
      v19 = sub_3007260((__int64)&v26);
      v30 = v19;
      v31 = v20;
    }
    v29 = v20;
    v28 = v19;
    v21 = sub_CA1930(&v28);
    v34[0] = sub_2E7BD70(v14, 0x31u, v21 >> 3, v25, (int)v34, 0, v32, v33, 1u, 0, 0);
    sub_33E4DA0(a1, v16, v34, 1);
  }
  if ( (_WORD)v26 == v12 && (v12 || !v27) )
    return v16;
  else
    return sub_33FB4C0(a1, v16, 0, a2, v12, 0);
}

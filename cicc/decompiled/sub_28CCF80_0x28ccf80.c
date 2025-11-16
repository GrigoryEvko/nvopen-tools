// Function: sub_28CCF80
// Address: 0x28ccf80
//
__int64 __fastcall sub_28CCF80(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r13d
  __int64 v5; // r12
  __int64 v7; // rcx
  __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // r14
  _QWORD *v11; // r8
  __int64 v12; // r15
  int v13; // r11d
  _QWORD *v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // eax
  int v19; // eax
  char v20; // al
  __int64 *v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-60h]
  _QWORD *v23; // [rsp+8h] [rbp-58h]
  int v24; // [rsp+8h] [rbp-58h]
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  int v26; // [rsp+10h] [rbp-50h]
  _QWORD *v27; // [rsp+10h] [rbp-50h]
  int v28; // [rsp+10h] [rbp-50h]
  _QWORD *v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  _QWORD *v31; // [rsp+18h] [rbp-48h]
  _QWORD *v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  _QWORD *v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+28h] [rbp-38h]
  _QWORD *v37; // [rsp+28h] [rbp-38h]
  _QWORD *v38; // [rsp+28h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 24);
  if ( v3 )
  {
    v5 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    LODWORD(v8) = *(_QWORD *)(*a2 + 16);
    if ( !(_DWORD)v8 )
    {
      v32 = a3;
      v36 = *(_QWORD *)(a1 + 8);
      v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 32LL))(v5);
      a3 = v32;
      v7 = v36;
      *(_QWORD *)(v5 + 16) = v8;
      v5 = *a2;
    }
    v9 = v3 - 1;
    v10 = v9 & (unsigned int)v8;
    v11 = (_QWORD *)(v7 + 56 * v10);
    v12 = *v11;
    if ( *v11 != v5 )
    {
      v13 = 1;
      v14 = 0;
      while ( 1 )
      {
        if ( v5 != -8 && v5 != 0x7FFFFFFF0LL && v12 != 0x7FFFFFFF0LL && v12 != -8 )
        {
          v15 = *(_QWORD *)(v12 + 16);
          if ( !(_DWORD)v15 )
          {
            v23 = a3;
            v26 = v13;
            v29 = v11;
            v33 = v7;
            v37 = v14;
            v16 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD *, __int64))(*(_QWORD *)v12 + 32LL))(
                    v12,
                    v15,
                    a3,
                    v7,
                    v11,
                    0x7FFFFFFF0LL);
            a3 = v23;
            v13 = v26;
            v11 = v29;
            v7 = v33;
            *(_QWORD *)(v12 + 16) = v16;
            v15 = v16;
            v14 = v37;
          }
          v17 = *(_QWORD *)(v5 + 16);
          if ( !(_DWORD)v17 )
          {
            v22 = a3;
            v24 = v13;
            v27 = v11;
            v30 = v7;
            v34 = v14;
            v17 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD *, __int64))(*(_QWORD *)v5 + 32LL))(
                    v5,
                    v15,
                    a3,
                    v7,
                    v11,
                    0x7FFFFFFF0LL);
            a3 = v22;
            v13 = v24;
            v11 = v27;
            v7 = v30;
            *(_QWORD *)(v5 + 16) = v17;
            v14 = v34;
          }
          if ( v15 == v17 )
          {
            v18 = *(_DWORD *)(v5 + 12);
            if ( v18 == *(_DWORD *)(v12 + 12) )
            {
              if ( v18 > 0xFFFFFFFD )
                break;
              v19 = *(_DWORD *)(v5 + 8);
              if ( (unsigned int)(v19 - 11) <= 1 || v19 == *(_DWORD *)(v12 + 8) )
              {
                v25 = a3;
                v28 = v13;
                v31 = v11;
                v35 = v7;
                v38 = v14;
                v20 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, __int64, _QWORD *, __int64))(*(_QWORD *)v5 + 16LL))(
                        v5,
                        v12,
                        a3,
                        v7,
                        v11,
                        0x7FFFFFFF0LL);
                v14 = v38;
                v7 = v35;
                v11 = v31;
                v13 = v28;
                a3 = v25;
                if ( v20 )
                  break;
              }
            }
          }
        }
        if ( *v11 == -8 )
        {
          if ( !v14 )
            v14 = v11;
          *a3 = v14;
          return 0;
        }
        if ( v14 || *v11 != 0x7FFFFFFF0LL )
          v11 = v14;
        v5 = *a2;
        LODWORD(v10) = v9 & (v13 + v10);
        v21 = (__int64 *)(v7 + 56LL * (unsigned int)v10);
        v12 = *v21;
        if ( *v21 == *a2 )
        {
          v11 = (_QWORD *)(v7 + 56LL * (unsigned int)v10);
          break;
        }
        v14 = v11;
        ++v13;
        v11 = (_QWORD *)(v7 + 56LL * (unsigned int)v10);
      }
    }
    *a3 = v11;
    return 1;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
